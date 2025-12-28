# minitune/rl/dpo.py
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.progress import Progress
from pathlib import Path

from ..config import TrainConfig
from ..data import dpo_collate_fn
from ..utils.profiling import MFUCalculator

class DPOTrainer:
    def __init__(self, config: TrainConfig, train_dataset):
        self.config = config
        self.accelerator = Accelerator(log_with="tensorboard", project_dir=config.dpo.output_dir)
        
        # Load Components
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Policy Model (Trainable)
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            attn_implementation="flash_attention_2" if config.model.use_flash_attention_2 else "default",
            torch_dtype=torch.bfloat16,
            # device_map=self.accelerator.device, # device_map=self.accelerator.device forces the entire model onto the GPU before accelerate can shard it. This is not useful
            trust_remote_code=True,
        )
        if config.peft:
            peft_conf = LoraConfig(**config.peft.__dict__)
            self.policy_model = get_peft_model(model, peft_conf)
            self.policy_model = self.policy_model.to(torch.bfloat16) # Fix dtype mismatch between base model and LoRA layers
        else:
            self.policy_model = model

        # Reference Model (Frozen)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            attn_implementation="flash_attention_2" if config.model.use_flash_attention_2 else "default",
            torch_dtype=torch.bfloat16,
            # device_map=self.accelerator.device, 
            trust_remote_code=True,
        )
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)

        # Optimizer
        self.optimizer = AdamW(self.policy_model.parameters(), lr=config.dpo.learning_rate)

        # Dataloader
        from torch.utils.data import DataLoader
        self.dataloader = DataLoader(
            train_dataset, 
            batch_size=config.dpo.batch_size, 
            collate_fn=dpo_collate_fn, # Use custom collator
            shuffle=True,
            num_workers=4,        # Use CPU cores to prep data
            pin_memory=True,      # Faster RAM->VRAM transfer
            prefetch_factor=2,    # Pre-load 2 batches per worker
            persistent_workers=True # Don't kill workers after every epoch
        )

        # --- 1. Init Profiler ---
        self.mfu_calc = MFUCalculator(self.policy_model) # doing it before sharding (`accelerator.prepare`) to get correct param count
        
        # Prepare
        self.policy_model, self.ref_model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.policy_model, self.ref_model, self.optimizer, self.dataloader
        )
        
        if self.accelerator.is_main_process:
             from torch.utils.tensorboard import SummaryWriter
             self.writer = SummaryWriter(log_dir=f"{config.dpo.output_dir}/logs")

    def get_batch_logps(self, model, input_ids, attention_mask, labels):
        """
        Computes the log probabilities of the 'labels' tokens.
        Includes masking logic to ignore prompt tokens (where labels == -100).
        """
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift Logic for Causal LM:
        # logits[i] predicts input_ids[i+1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute Log Softmax
        # shape: [batch, seq_len-1, vocab]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # 1. Create a safe version of labels for gathering (torch.gather)
        # We replace -100 with 0 (or any valid index) so gather doesn't crash
        labels_for_gather = shift_labels.clone()
        labels_for_gather[labels_for_gather == -100] = 0
        
        # 2. Gather the log-prob of the target token
        # shape: [batch, seq_len-1]
        token_log_probs = torch.gather(log_probs, -1, labels_for_gather.unsqueeze(-1)).squeeze(-1)
        
        # 3. Masking: Ignore tokens where real label is -100 (Prompt or Padding)
        mask = (shift_labels != -100).float()
        
        # 4. Sum log probs over valid tokens only
        # The '0' we gathered earlier gets multiplied by 0 here, so it's ignored.
        sum_log_probs = (token_log_probs * mask).sum(-1)
        
        return sum_log_probs

    def train(self, save_model=False):
        self.policy_model.train()
        progress = Progress(disable=not self.accelerator.is_main_process)
        task = progress.add_task("DPO Training...", total=len(self.dataloader)*self.config.dpo.epochs)
        global_step = 0
        
        accumulated_tokens = 0 # for MFU calculation
        
        with progress:
            for epoch in range(self.config.dpo.epochs):
                for batch in self.dataloader:
                    with self.accelerator.accumulate(self.policy_model):
                        
                        def compute_logps(model, prefix):
                            return self.get_batch_logps(
                                model, 
                                batch[f"{prefix}_input_ids"], 
                                batch[f"{prefix}_attention_mask"], 
                                batch[f"{prefix}_labels"]
                            )

                        policy_chosen_logps = compute_logps(self.policy_model, "chosen")
                        policy_rejected_logps = compute_logps(self.policy_model, "rejected")

                        with torch.no_grad():
                            ref_chosen_logps = compute_logps(self.ref_model, "chosen")
                            ref_rejected_logps = compute_logps(self.ref_model, "rejected")

                        pi_logratios = policy_chosen_logps - policy_rejected_logps
                        ref_logratios = ref_chosen_logps - ref_rejected_logps
                        
                        logits = pi_logratios - ref_logratios
                        
                        losses = -F.logsigmoid(self.config.dpo.beta * logits)
                        loss = losses.mean()

                        self.accelerator.backward(loss)
                        
                        #Count tokens for every micro-batch (for MFU calculation)
                        accumulated_tokens += batch["chosen_input_ids"].numel()
                        accumulated_tokens += batch["rejected_input_ids"].numel()
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        # --- 2. Calculate MFU ---
                        mfu, tps = self.mfu_calc.step(accumulated_tokens)
                        accumulated_tokens = 0  # Reset for next accumulation cycle

                        # Update Progress Bar with MFU
                        progress.update(task, advance=1, description=f"DPO (MFU: {mfu*100:.2f}%)")
                        
                        global_step += 1
                        
                        with torch.no_grad():
                            chosen_rewards = self.config.dpo.beta * (policy_chosen_logps - ref_chosen_logps).mean()
                            rejected_rewards = self.config.dpo.beta * (policy_rejected_logps - ref_rejected_logps).mean()
                            accuracy = (policy_chosen_logps > policy_rejected_logps).float().mean()

                        if self.accelerator.is_main_process and global_step % self.config.dpo.logging_steps == 0:
                            self.writer.add_scalar("loss/dpo", loss.item(), global_step)
                            self.writer.add_scalar("rewards/chosen", chosen_rewards.item(), global_step)
                            self.writer.add_scalar("rewards/rejected", rejected_rewards.item(), global_step)
                            self.writer.add_scalar("rewards/accuracy", accuracy.item(), global_step)
                            # Log Performance
                            self.writer.add_scalar("perf/mfu", mfu, global_step)
                            self.writer.add_scalar("perf/tokens_per_sec", tps, global_step)
                            
        if save_model: self.save_model()

    def save_model(self):
        if self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            # Save Policy Model only
            unwrapped = self.accelerator.unwrap_model(self.policy_model)
            path = Path(self.config.dpo.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            unwrapped.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"DPO Model saved to {path}")