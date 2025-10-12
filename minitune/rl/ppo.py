import torch
import torch.nn.functional as F
from accelerate import Accelerator
from peft import PeftModel, LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from rich.progress import Progress
from pathlib import Path
from typing import List

from .core import RolloutBuffer
from ..config import TrainConfig
from ..data import load_and_prepare_dataset

class PPOTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.accelerator = Accelerator(log_with="tensorboard", project_dir=config.rl.output_dir)

        # Setup logging
        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(log_dir=f"{config.rl.output_dir}/logs")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" # For generation

        # Load SFT model as the policy model
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            use_flash_attention_2=config.model.use_flash_attention_2,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.policy_model = PeftModel.from_pretrained(base_model, config.rl.sft_model_path)

        # Load reference model (frozen SFT model)
        self.ref_model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                config.model.name_or_path,
                use_flash_attention_2=config.model.use_flash_attention_2,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ),
            config.rl.sft_model_path
        )
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Load reward model
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.rl.reward_model_path
        )
        for param in self.reward_model.parameters():
            param.requires_grad = False

        self.optimizer = AdamW(self.policy_model.parameters(), lr=config.rl.learning_rate)

        (
            self.policy_model,
            self.ref_model,
            self.reward_model,
            self.optimizer,
        ) = self.accelerator.prepare(
            self.policy_model, self.ref_model, self.reward_model, self.optimizer
        )
        self.rollout_buffer = RolloutBuffer()
        self.global_step = 0

    def train(self):
        """Main PPO training loop: Rollout -> Learning."""
        prompts_dataset = load_and_prepare_dataset(self.config.data, self.tokenizer)
        prompts_dataloader = DataLoader(prompts_dataset, batch_size=self.config.rl.batch_size)
        prompts_dataloader = self.accelerator.prepare(prompts_dataloader)
        
        progress = Progress(disable=not self.accelerator.is_main_process)
        rollout_task = progress.add_task("[cyan]Rollouts...", total=self.config.rl.num_rollouts)
        
        with progress:
            for rollout in range(self.config.rl.num_rollouts):
                # --- 1. Rollout Phase ---
                self.rollout_buffer.clear()
                self.policy_model.eval()
                for batch in prompts_dataloader:
                    prompt_tensors = batch["input_ids"]
                    
                    # Generate responses from policy model
                    responses = self.accelerator.unwrap_model(self.policy_model).generate(
                        input_ids=prompt_tensors,
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=self.config.rl.max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                    )
                    
                    # Compute logprobs, rewards
                    with torch.no_grad():
                        logprobs, ref_logprobs = self.compute_logprobs(prompt_tensors, responses)
                        rewards = self.compute_rewards(responses)
                        kl_penalty = logprobs - ref_logprobs
                        rewards -= self.config.rl.kl_penalty_coeff * kl_penalty
                    
                    for i in range(len(prompt_tensors)):
                        self.rollout_buffer.add(prompt_tensors[i], responses[i], logprobs[i], rewards[i], torch.tensor(0.0))

                progress.update(rollout_task, advance=1)
                
                # --- 2. Learning Phase ---
                self.learn()
        self.save_model()

    def compute_logprobs(self, prompts, responses):
        """Compute log probabilities of responses given prompts."""
        # This is a simplified implementation. A real one would be more careful with padding.
        full_text = torch.cat([prompts, responses], dim=-1)
        full_attention_mask = (full_text != self.tokenizer.pad_token_id).long()
        
        with torch.no_grad():
            ref_outputs = self.ref_model(full_text, attention_mask=full_attention_mask)
            ref_logits = ref_outputs.logits
            ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        
        policy_outputs = self.policy_model(full_text, attention_mask=full_attention_mask)
        policy_logits = policy_outputs.logits
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        
        return policy_logprobs, ref_logprobs
    
    def compute_rewards(self, responses):
        """Compute rewards using the reward model."""
        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        # Dummy reward computation; replace with actual reward model forward pass
        # For simplicity, returning random rewards.
        return torch.randn(len(response_texts), device=self.accelerator.device)

    def learn(self):
        """Update policy using PPO."""
        self.policy_model.train()
        for _ in range(self.config.rl.ppo_epochs):
            for prompts, responses, old_logprobs, rewards, _ in self.rollout_buffer:
                # In a full PPO impl, you'd calculate advantages and use a value function.
                # Here, we simplify and use rewards directly as advantages.
                
                full_text = torch.cat([prompts.unsqueeze(0), responses.unsqueeze(0)], dim=-1)
                attention_mask = (full_text != self.tokenizer.pad_token_id).long()
                
                outputs = self.policy_model(full_text, attention_mask=attention_mask)
                logprobs = F.log_softmax(outputs.logits, dim=-1)
                
                ratio = torch.exp(logprobs - old_logprobs)
                
                # PPO Clipped Surrogate Objective (simplified)
                policy_loss_1 = ratio * rewards
                policy_loss_2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * rewards
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                self.accelerator.backward(policy_loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.accelerator.is_main_process:
                    self.writer.add_scalar("loss/ppo", policy_loss.item(), self.global_step)
                self.global_step += 1

    def save_model(self):
        """Saves the final policy model."""
        if self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.policy_model)
            
            output_path = Path(self.config.rl.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            unwrapped_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            print(f"Model saved to {output_path}")