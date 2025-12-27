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

class SFTTrainer:
    """Trainer for Supervised Fine-Tuning."""

    def __init__(
        self, 
        config: TrainConfig, 
        train_dataset: Optional[Dataset] = None, 
        loss_fn: Optional[nn.Module] = None
    ):
        self.config = config
        self.loss_fn = loss_fn # Store the custom loss function
        self.accelerator = Accelerator(log_with="tensorboard", project_dir=config.sft.output_dir)

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
            use_flash_attention_2=config.model.use_flash_attention_2,
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

    def train(self):
        """Main training loop."""
        total_steps = len(self.train_dataloader) // self.config.sft.gradient_accumulation_steps * self.config.sft.epochs
        
        progress = Progress(disable=not self.accelerator.is_main_process)
        task = progress.add_task("[green]Training...", total=total_steps)

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
                        progress.update(task, advance=1)
                        self.global_step += 1
                        if self.global_step % self.config.sft.logging_steps == 0:
                            if self.accelerator.is_main_process:
                                self.writer.add_scalar("loss/sft", loss.item(), self.global_step)
        
        self.save_model()

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