# minitune/sft.py
import torch
import torch.nn as nn
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from rich.progress import Progress
from pathlib import Path
from typing import Optional, Union
from dataclasses import asdict, is_dataclass
from omegaconf import DictConfig, OmegaConf

from .config import TrainConfig
from .data import load_and_prepare_dataset
from .utils.profiling import MFUCalculator

class SFTTrainer:
    """Trainer for Supervised Fine-Tuning."""

    def __init__(
        self, 
        config: TrainConfig, 
        train_dataset: Optional[Dataset] = None, 
        eval_dataset: Optional[Dataset] = None,
        loss_fn: Optional[nn.Module] = None,
        eval_fn: Optional[callable] = None
    ):
        self.config = config
        self.loss_fn = loss_fn # Store the custom loss function
        self.accelerator = Accelerator(log_with="tensorboard", project_dir=config.sft.output_dir)

        # Store the hooks
        self.eval_dataset = eval_dataset
        self.eval_fn = eval_fn
        
        # Setup logging
        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(log_dir=f"{config.sft.output_dir}/logs")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # If dataset is passed programmatically, use it. Otherwise load from config.
        if train_dataset is not None:
            self.train_dataset = train_dataset
        else:
            self.train_dataset = load_and_prepare_dataset(config.data, self.tokenizer)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.sft.batch_size,
            collate_fn=data_collator,
            shuffle=True
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            #use_flash_attention_2=config.model.use_flash_attention_2,
            attn_implementation="flash_attention_2" if config.model.use_flash_attention_2 else "default",
            torch_dtype=torch.bfloat16,
            device_map=self.accelerator.device, 
            trust_remote_code=True,
        )

        # --- Safe PEFT Config Loading (in accordance with OmegaConf)---
        if config.peft:
            # We need to convert config.peft to a clean dict, handling 
            # both Dataclasses and OmegaConf DictConfigs
            if isinstance(config.peft, DictConfig):
                # Resolve OmegaConf object to primitive dict (removes _metadata)
                peft_args = OmegaConf.to_container(config.peft, resolve=True)
            elif is_dataclass(config.peft):
                peft_args = asdict(config.peft)
            else:
                # It's already a dict (fallback)
                peft_args = config.peft if isinstance(config.peft, dict) else config.peft.__dict__

            # Filter out any internal keys that might still persist (safety net)
            peft_args = {k: v for k, v in peft_args.items() if not k.startswith("_")}
            
            peft_config = LoraConfig(**peft_args)
            self.model = get_peft_model(model, peft_config)
            
            # Force LoRA adapters (often fp32) to match base model (bf16), else while distributed training when communication happens, lora weights and base model weights have different dtypes
            # This prevents "ValueError: Must flatten tensors with uniform dtype" in DDP/FSDP
            if config.model.name_or_path: 
                 # We assume generic bf16 training based on your previous configs
                 self.model = self.model.to(torch.bfloat16)
        else:
            self.model = model

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.sft.learning_rate)

        # Prepare for distributed training
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )
        self.global_step = 0

    def compute_loss(self, model, batch):
        """
        Helper to compute loss, using custom function if provided.
        """
        outputs = model(**batch)
        
        if self.loss_fn is not None:
            # Shift labels/logits for Causal LM (Predict Next Token)
            # Logits: [Batch, Seq, Vocab] -> Remove last token prediction
            logits = outputs.logits[..., :-1, :].contiguous()
            # Labels: [Batch, Seq] -> Remove first token (since it wasn't predicted)
            labels = batch["labels"][..., 1:].contiguous()
            
            return self.loss_fn(logits, labels)
            
        return outputs.loss

    def train(self, save_model=False):
        """Main training loop."""
        total_steps = len(self.train_dataloader) // self.config.sft.gradient_accumulation_steps * self.config.sft.epochs
        
        progress = Progress(disable=not self.accelerator.is_main_process)
        task = progress.add_task("[green]Training...", total=total_steps)

        # --- Initialize Profiler ---
        mfu_calc = MFUCalculator(self.model)

        with progress:
            for epoch in range(self.config.sft.epochs):
                for step, batch in enumerate(self.train_dataloader):
                    self.model.train()
                    with self.accelerator.accumulate(self.model):
                        loss = self.compute_loss(self.model, batch)
                        self.accelerator.backward(loss)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        # --- Profiling Step ---
                        # Calculate total tokens in this batch (Batch Size * Seq Len)
                        # Note: If we use gradient accumulation, this is technically 
                        # tokens per micro-step, but MFU averages out over time.
                        # Here, we calculate throughput per physical forward/backward pass.
                        
                        # We just finished 'gradient_accumulation_steps' micro-batches.
                        # The timer covers that whole duration.
                        # So we must sum the tokens from ALL those steps.
                        
                        # Current batch tokens
                        micro_batch_tokens = batch["input_ids"].numel()
                        
                        # Total tokens processed during this time window
                        total_tokens_accumulated = micro_batch_tokens * self.config.sft.gradient_accumulation_steps
                        
                        mfu, tps = mfu_calc.step(total_tokens_accumulated)

                        progress.update(task, advance=1, description=f"[green]SFT (MFU: {mfu*100:.2f}%)")
                        self.global_step += 1
                        
                        if self.global_step % self.config.sft.logging_steps == 0:
                            if self.accelerator.is_main_process:
                                self.writer.add_scalar("loss/sft", loss.item(), self.global_step)
                                self.writer.add_scalar("perf/mfu", mfu, self.global_step)
                                self.writer.add_scalar("perf/tokens_per_sec", tps, self.global_step)
                        
                        if self.eval_fn and self.global_step % self.config.sft.eval_steps == 0:
                            # 1. Sync before eval
                            self.accelerator.wait_for_everyone()
                            
                            # 2. Run the user-provided function
                            # We pass 'self' so the function has access to model/tokenizer
                            metrics = self.eval_fn(self) 
                            
                            # 3. Log results (Only Main Process writes to disk)
                            if self.accelerator.is_main_process and metrics:
                                for metric_name, value in metrics.items():
                                    self.writer.add_scalar(f"eval/{metric_name}", value, self.global_step)
                                    # Optional: Print to console
                                    progress.console.print(f"[bold blue]Eval step {self.global_step}: {metric_name} = {value:.4f}[/bold blue]")
                            
                            # 4. Sync after eval to prevent race conditions
                            self.accelerator.wait_for_everyone()
        
        if save_model: self.save_model()

    def save_model(self):
        """Saves the final model."""
        if self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            
            output_path = Path(self.config.sft.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            unwrapped_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            print(f"Model saved to {output_path}")