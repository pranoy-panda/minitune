# minitune/examples/04_run_sft_with_profiling.py
# $ accelerate launch --gpu_ids 0,1,2,3 --config_file configs/fsdp_a40.yaml examples/04_run_sft_with_profiling.py
import typer
import torch
import math
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from rich.console import Console
from rich.table import Table

# Internal imports
from minitune.config import TrainConfig, ModelConfig, DataConfig, SFTConfig, PeftConfig
from minitune.sft import SFTTrainer
from minitune.data import load_and_prepare_dataset
from minitune.losses import FocalLoss

console = Console()

def evaluate_perplexity(trainer, eval_dataset, num_batches=30):
    """
    Computes perplexity on a subset of the validation set.
    """
    if trainer.accelerator.is_main_process:
        console.print(f"[bold cyan]Running Evaluation (Max {num_batches} batches)...[/bold cyan]")
    model = trainer.model
    model.eval()
    
    collate_fn = DataCollatorForLanguageModeling(trainer.tokenizer, mlm=False)
    eval_loader = DataLoader(eval_dataset, batch_size=trainer.config.sft.batch_size, collate_fn=collate_fn)
    
    total_loss = 0
    steps = 0
    
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= num_batches: break
            # Move to device
            batch = {k: v.to(trainer.accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            steps += 1
            
    if steps == 0: return float('inf')
    
    avg_loss = total_loss / steps
    return math.exp(avg_loss)

def compute_metrics(trainer, eval_set_samples=50):
    """
    This is the function handle we will pass to the trainer.
    It computes perplexity on the eval_dataset.
    """
    # Grab the eval dataset stored in the trainer
    if trainer.eval_dataset is None:
        return {}
        
    eval_subset = trainer.eval_dataset.select(range(min(eval_set_samples, len(trainer.eval_dataset))))
    
    collate_fn = DataCollatorForLanguageModeling(trainer.tokenizer, mlm=False)
    loader = DataLoader(eval_subset, batch_size=trainer.config.sft.batch_size, collate_fn=collate_fn)
    loader = trainer.accelerator.prepare(loader) # Prepare for distributed training
    
    trainer.model.eval()
    total_loss = 0
    steps = 0
    
    with torch.no_grad():
        for batch in loader:
            outputs = trainer.model(**batch)
            total_loss += outputs.loss.item()
            steps += 1
            
    trainer.model.train() # Switch back to train mode
    
    if steps == 0: return {}
    
    # Calculate metrics
    avg_loss = total_loss / steps
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float('inf')
        
    return {"perplexity": perplexity, "loss": avg_loss}

def main(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    dataset_name: str = "HuggingFaceH4/no_robots", 
    dataset_format: str = "chat",     # or "instruction"
    output_dir: str = "./outputs/profiling_run"
):
    # 1. Config Setup (Programmatic)
    config = TrainConfig(
        model=ModelConfig(
            name_or_path=model_name, 
            use_flash_attention_2=True
        ),
        data=DataConfig(
            path=dataset_name,
            format_type=dataset_format,
            # no_robots uses "messages", alpaca uses "instruction"/"output"
            chat_column="messages",   
            prompt_column="instruction", 
            response_column="response",
            test_size=0.1,
            max_input_token_length=1024 # Keep 1024 for speed, 4096 for real runs
        ),
        peft=PeftConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ),
        sft=SFTConfig(
            output_dir=output_dir,
            batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            epochs=1,
            logging_steps=5
        )
    )

    # 2. Initialize Trainer
    console.print(f"[bold green]Initializing Model: {model_name}[/bold green]")
    trainer = SFTTrainer(config)

    # 3. Load & Process Data
    # Returns tuple (train, test) because test_size=0.1
    train_data, test_data = load_and_prepare_dataset(config.data, trainer.tokenizer, split="train")
    
    # using a subset
    train_data, test_data = train_data.select(range(2000)), test_data.select(range(500))
    
    print(f"Train Dataset Size: {len(train_data)} \n Test Dataset Size: {len(test_data)}")
    
    # Inject Data into Trainer
    trainer.train_dataset = train_data
    trainer.eval_dataset = test_data
    trainer.eval_fn = compute_metrics
    
    # Refresh the dataloader to use the new dataset
    trainer.train_dataloader = DataLoader(
        train_data,
        batch_size=config.sft.batch_size,
        collate_fn=DataCollatorForLanguageModeling(trainer.tokenizer, mlm=False),
        shuffle=True
    )
    trainer.train_dataloader = trainer.accelerator.prepare(trainer.train_dataloader)

    # 4. Inject Custom Loss (Focal Loss)
    # Good for instruction tuning to focus on hard tokens
    trainer.loss_fn = FocalLoss(gamma=2.0)

    # 5. Experiment Execution
    
    # Phase A: Baseline Eval
    trainer.accelerator.wait_for_everyone() # Sync before starting
    if trainer.accelerator.is_main_process: # log only on main process
        console.rule("[bold red]Phase 1: Before SFT eval[/bold red]")
    
    # The model needs all GPUs to synchronize to calculate loss
    ppl_before = evaluate_perplexity(trainer, test_data)

    if trainer.accelerator.is_main_process: # log only on main process
        console.print(f"Perplexity BEFORE: [bold red]{ppl_before:.2f}[/bold red]")

    # Phase B: Training
    # (Everyone runs training, so NO is_main_process check here)
    if trainer.accelerator.is_main_process:
        console.rule("[bold green]Phase 2: Training & Profiling[/bold green]")
    trainer.train()

    # Phase C: Post-Eval
    trainer.accelerator.wait_for_everyone() # Sync after training
    if trainer.accelerator.is_main_process:
        console.rule("[bold blue]Phase 3: Post SFT Eval[/bold blue]")
    
    ppl_after = evaluate_perplexity(trainer, test_data)
    
    if trainer.accelerator.is_main_process:
        # 6. Report
        table = Table(title=f"Experiment Results: {model_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Before SFT", style="red")
        table.add_column("After SFT", style="green")
        table.add_column("Improvement", style="yellow")
        
        delta = ppl_before - ppl_after
        table.add_row("Perplexity", f"{ppl_before:.2f}", f"{ppl_after:.2f}", f"{delta:.2f}")
        
        console.print(table)
        console.print(f"Run saved to: {output_dir}")

if __name__ == "__main__":
    typer.run(main)