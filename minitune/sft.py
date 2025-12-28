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
from typing import Optional, Union, Dict
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
            # Handle the case where loader returns (train, test) tuple
            loaded_data = load_and_prepare_dataset(config.data, self.tokenizer)
            if isinstance(loaded_data, tuple):
                self.train_dataset = loaded_data[0]
                # Automatically set eval_dataset if it wasn't provided explicitly
                if self.eval_dataset is None:
                    self.eval_dataset = loaded_data[1]
            else:
                self.train_dataset = loaded_data

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.sft.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            num_workers=4,        # Use CPU cores to prep data
            pin_memory=True,      # Faster RAM->VRAM transfer
            prefetch_factor=2,    # Pre-load 2 batches per worker
            persistent_workers=True # Don't kill workers after every epoch
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            attn_implementation="flash_attention_2" if config.model.use_flash_attention_2 else "default",
            torch_dtype=torch.bfloat16,
            device_map=None, # Explicitly None for FSDP/Accelerator handling
            trust_remote_code=True,
        )

        # --- Safe PEFT Config Loading ---
        if config.peft:
            if isinstance(config.peft, DictConfig):
                peft_args = OmegaConf.to_container(config.peft, resolve=True)
            elif is_dataclass(config.peft):
                peft_args = asdict(config.peft)
            else:
                peft_args = config.peft if isinstance(config.peft, dict) else config.peft.__dict__

            peft_args = {k: v for k, v in peft_args.items() if not k.startswith("_")}
            
            peft_config = LoraConfig(**peft_args)
            self.model = get_peft_model(model, peft_config)
            
            # Force LoRA adapters to bf16 for FSDP/DDP compatibility
            if config.model.name_or_path: 
                 self.model = self.model.to(torch.bfloat16)
        else:
            self.model = model

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.sft.learning_rate)

        # --- Initialize Profiler ---
        # Initialize before prepare() to get accurate parameter count
        self.mfu_calc = MFUCalculator(self.model)
        
        # Prepare for distributed training
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )
        self.global_step = 0

    def compute_loss(self, model, batch):
        """Helper to compute loss, using custom function if provided."""
        outputs = model(**batch)
        
        if self.loss_fn is not None:
            logits = outputs.logits[..., :-1, :].contiguous()
            labels = batch["labels"][..., 1:].contiguous()
            return self.loss_fn(logits, labels)
            
        return outputs.loss

    def train(self, save_model=False, max_steps: Optional[int] = None) -> Dict[str, float]:
        """
        Main training loop.
        
        Args:
            save_model: Whether to save the final model to disk.
            max_steps: If provided, stops training after this many steps (used for Auto-Tuning).
            
        Returns:
            Dict containing profiling metrics (MFU, Peak Memory, etc.)
        """
        # Calculate total optimization steps (Updates)
        total_steps = (len(self.train_dataloader) // self.config.sft.gradient_accumulation_steps) * self.config.sft.epochs
        
        if max_steps:
            total_steps = min(total_steps, max_steps)
        
        progress = Progress(disable=not self.accelerator.is_main_process)
        task = progress.add_task("[green]Training...", total=total_steps)
        
        accumulated_tokens = 0 
        avg_mfu = 0.0
        steps_counted = 0
        
        with progress:
            for epoch in range(self.config.sft.epochs):
                for step, batch in enumerate(self.train_dataloader):
                    
                    # Stop if we hit the limit
                    if max_steps and self.global_step >= max_steps:
                        break
                        
                    self.model.train()
                    with self.accelerator.accumulate(self.model):
                        loss = self.compute_loss(self.model, batch)
                        self.accelerator.backward(loss)
                        accumulated_tokens += batch["input_ids"].numel()
                    
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        # --- Profiling & Update Step ---
                        mfu, tps = self.mfu_calc.step(accumulated_tokens)
                        accumulated_tokens = 0 
                        
                        if self.global_step > 5:
                            avg_mfu += mfu
                            steps_counted += 1

                        # Update Progress Bar
                        # progress.update(task, advance=1, description=f"[green]SFT (MFU: {mfu*100:.2f}%)")
                        progress.update(
                            task, 
                            advance=1, 
                            description=f"[green]Epoch {epoch+1}/{self.config.sft.epochs} | Step {self.global_step}/{total_steps} | MFU: {mfu*100:.2f}%"
                        )
                        self.global_step += 1
                        
                        # Logging
                        if self.global_step % self.config.sft.logging_steps == 0:
                            if self.accelerator.is_main_process:
                                self.writer.add_scalar("loss/sft", loss.item(), self.global_step)
                                self.writer.add_scalar("perf/mfu", mfu, self.global_step)
                                self.writer.add_scalar("perf/tokens_per_sec", tps, self.global_step)
                        
                        # Evaluation
                        if self.eval_fn and self.global_step % self.config.sft.eval_steps == 0:
                            self.accelerator.wait_for_everyone()
                            metrics = self.eval_fn(self) 
                            if self.accelerator.is_main_process and metrics:
                                for metric_name, value in metrics.items():
                                    self.writer.add_scalar(f"eval/{metric_name}", value, self.global_step)
                            self.accelerator.wait_for_everyone()
                
                if max_steps and self.global_step >= max_steps:
                    break
        
        if save_model: self.save_model()
        
        return {
            "avg_mfu": avg_mfu / max(1, steps_counted),
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        }

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