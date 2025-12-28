# minitune/minitune/autotuner/trial.py
import sys
import json
import typer
import torch
import traceback
from pathlib import Path

# Add project root to sys.path to ensure imports work
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from minitune.config import TrainConfig, ModelConfig, DataConfig, SFTConfig, PeftConfig
from minitune.sft import SFTTrainer

app = typer.Typer()

@app.command()
def run_trial(
    model_name: str = typer.Option(..., help="Model name or path"),
    dataset_path: str = typer.Option(..., help="Path to dataset"),
    batch_size: int = typer.Option(..., help="Micro-batch size"),
    grad_accum_steps: int = typer.Option(..., help="Gradient accumulation steps"),
    output_file: str = typer.Option(..., help="Path to save result JSON"),
    max_steps: int = typer.Option(50, help="Number of profiling steps"),
    use_flash_attn: bool = typer.Option(True, help="Use Flash Attention 2"),
):
    """
    Executes a single training trial.
    If successful, writes metrics to output_file.
    If OOM, writes error status to output_file.
    """
    try:
        # 1. Construct Configuration
        config = TrainConfig(
            model=ModelConfig(
                name_or_path=model_name,
                use_flash_attention_2=use_flash_attn
            ),
            data=DataConfig(
                path=dataset_path,
                format_type="chat", # Defaulting to generic chat for tuning
                split="train",
                max_input_token_length=1024 # Standard tuning length
            ),
            peft=PeftConfig(r=8, lora_alpha=16),
            sft=SFTConfig(
                output_dir="./tmp_autotune", # Temporary dir
                batch_size=batch_size,
                gradient_accumulation_steps=grad_accum_steps,
                learning_rate=2e-5,
                epochs=1,
                logging_steps=5
            )
        )

        print(f"[Trial] Starting: Batch={batch_size}, Accum={grad_accum_steps}")
        
        # 2. Initialize Trainer
        trainer = SFTTrainer(config)
        
        # 3. Run Dry Run
        # This will train for 'max_steps' and then return average MFU
        metrics = trainer.train(save_model=False, max_steps=max_steps)
        
        trainer.accelerator.wait_for_everyone() # Ensure all ranks finish
        if trainer.accelerator.is_main_process:
            result = {
                "status": "success",
                "mfu": metrics["avg_mfu"],
                "peak_memory_mb": metrics["peak_memory_mb"],
                "config": {
                    "batch_size": batch_size,
                    "grad_accum_steps": grad_accum_steps
                }
            }
            
            with open(output_file, "w") as f:
                json.dump(result, f)
            print(f"[Trial] Success! MFU={metrics['avg_mfu']:.4f}")

    except torch.cuda.OutOfMemoryError:
        print("[Trial] FAILED: CUDA Out of Memory")
        result = {
            "status": "oom",
            "config": {"batch_size": batch_size}
        }
        with open(output_file, "w") as f:
            json.dump(result, f)
        sys.exit(1) # Exit with error code

    except Exception as e:
        print(f"[Trial] FAILED: {str(e)}")
        traceback.print_exc()
        result = {
            "status": "error",
            "error_msg": str(e)
        }
        with open(output_file, "w") as f:
            json.dump(result, f)
        sys.exit(1)

if __name__ == "__main__":
    app()