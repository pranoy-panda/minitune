# minitune/examples/02_run_dpo.py
# $ accelerate launch --config_file configs/accelerate_fsdp_a40.yaml examples/02_run_dpo.py
import typer
import torch
import random
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from minitune.config import TrainConfig, ModelConfig, DataConfig, DPOConfig, PeftConfig
from minitune.rl.dpo import DPOTrainer
from minitune.data import prepare_dpo_data
from minitune.data import dpo_collate_fn

console = Console()

def evaluate_dpo_metrics(trainer, test_dataset, num_batches=20):
    if trainer.accelerator.is_main_process:
        console.print(f"[bold cyan]Running DPO Evaluation (Max {num_batches} batches)...[/bold cyan]")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=trainer.config.dpo.batch_size, 
        collate_fn=dpo_collate_fn, 
        shuffle=True
    )
    # This acts as a synchronization point! 
    # Everyone must reach here at roughly the same time.
    test_loader = trainer.accelerator.prepare(test_loader)
    
    trainer.policy_model.eval()
    
    local_correct = 0
    local_total = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_batches: break
            
            chosen_logps = trainer.get_batch_logps(
                trainer.policy_model, 
                batch["chosen_input_ids"], 
                batch["chosen_attention_mask"], 
                batch["chosen_labels"]
            )
            rejected_logps = trainer.get_batch_logps(
                trainer.policy_model, 
                batch["rejected_input_ids"], 
                batch["rejected_attention_mask"], 
                batch["rejected_labels"]
            )
            
            matches = (chosen_logps > rejected_logps)
            local_correct += matches.sum()
            local_total += matches.numel()

    # Convert to tensors for aggregation
    stats_tensor = torch.tensor([local_correct, local_total], device=trainer.accelerator.device)
    
    # Sum results from all GPUs
    dist_stats = trainer.accelerator.reduce(stats_tensor, reduction="sum")
    
    total_correct = dist_stats[0].item()
    total_samples = dist_stats[1].item()
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    trainer.policy_model.train()
    
    return accuracy

def main(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
    output_dir: str = "./outputs/dpo_run"
):
    # 1. Config
    config = TrainConfig(
        model=ModelConfig(name_or_path=model_name),
        data=DataConfig(
            path=dataset_name,
            format_type="preference",
            max_input_token_length=1024
        ),
        peft=PeftConfig(r=16, lora_alpha=32),
        dpo=DPOConfig(
            output_dir=output_dir,
            beta=0.1,
            learning_rate=5e-7,
            batch_size=1, 
            gradient_accumulation_steps=8,
            epochs=1,
            logging_steps=10
        )
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Prepare Data (Train & Test)
    # Load separate split for eval
    print("Preparing Data...")
    train_dataset = prepare_dpo_data(config.data, tokenizer, split="train_prefs[:5000]") 
    test_dataset = prepare_dpo_data(config.data, tokenizer, split="test_prefs[:200]") 
    
    # 3. Init Trainer
    trainer = DPOTrainer(config, train_dataset)
    
    # --- BEFORE TRAINING ---
    # Baseline Eval
    trainer.accelerator.wait_for_everyone()

    # Quantitative Check 1
    acc_before = evaluate_dpo_metrics(trainer, test_dataset) # needs all GPUs to synchronize to calculate loss
    
    if trainer.accelerator.is_main_process: # log only on main process
        console.print(f"Preference Accuracy BEFORE DPO training: [bold red]{acc_before:.2%}[/bold red]")
        console.rule("[bold green]Starting Training[/bold green]")
        
    # --- TRAIN ---
    trainer.train()
    
    # --- AFTER TRAINING ---
    trainer.accelerator.wait_for_everyone()
    
    # Quantitative Check 2
    acc_after = evaluate_dpo_metrics(trainer, test_dataset)
    
    # Quantitative Check 2 (Final Accuracy)
    if trainer.accelerator.is_main_process:
        console.print(f"Preference Accuracy AFTER DPO training: [bold red]{acc_after:.2%}[/bold red]")
        
        # Results Table
        table = Table(title="DPO Experiment Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Before", style="red")
        table.add_column("After", style="green")
        table.add_column("Delta", style="yellow")
        
        delta = acc_after - acc_before
        table.add_row("Preference Accuracy", f"{acc_before:.2%}", f"{acc_after:.2%}", f"{delta:+.2%}")
        console.print(table)

if __name__ == "__main__":
    typer.run(main)