import torch
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from rich.progress import Progress
from pathlib import Path

from .config import TrainConfig
from .data import load_and_prepare_dataset

class SFTTrainer:
    """Trainer for Supervised Fine-Tuning."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.accelerator = Accelerator(log_with="tensorboard", project_dir=config.sft.output_dir)

        # Setup logging
        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(log_dir=f"{config.sft.output_dir}/logs")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and prepare dataset
        train_dataset = load_and_prepare_dataset(config.data, self.tokenizer)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.sft.batch_size,
            collate_fn=data_collator
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            use_flash_attention_2=config.model.use_flash_attention_2,
            torch_dtype=torch.bfloat16,
            device_map=self.accelerator.device, # Let accelerate handle device placement
            trust_remote_code=True,
        )

        # Apply PEFT if configured
        if config.peft:
            peft_config = LoraConfig(**config.peft.__dict__)
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
                        outputs = self.model(**batch)
                        loss = outputs.loss
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