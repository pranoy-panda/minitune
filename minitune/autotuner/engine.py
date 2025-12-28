# minitune/minitune/autotuner/engine.py
'''
How to use?
$ python minitune/autotuner/engine.py   --model-name "Qwen/Qwen2.5-3B-Instruct"   --dataset-path "HuggingFaceH4/no_robots"   --target-global-batch-size 32   --num-gpus 4  --strategy-type "heuristic"
'''
import typer
import subprocess
import json
import os
import shutil
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import Optional

# Import our Strategy
from minitune.autotuner.strategies import HeuristicSearch, BayesianSearch

app = typer.Typer()
console = Console()

def run_subprocess_trial(
    model: str, 
    dataset: str, 
    batch_size: int, 
    accum: int, 
    result_file: str
):
    """
    Launches `minitune/autotuner/trial.py` via Accelerate.
    This creates a separate process, protecting the main tuner from CUDA crashes.
    """
    cmd = [
        "accelerate", "launch",
        "minitune/autotuner/trial.py",
        "--model-name", model,
        "--dataset-path", dataset,
        "--batch-size", str(batch_size),
        "--grad-accum-steps", str(accum),
        "--output-file", result_file,
        "--max-steps", "30" # Short run for profiling
    ]
    
    # Run silently (capture output) to keep UI clean, unless debug needed
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # If the python script crashed (not just OOM, but code error)
        # We manually write an error result if one doesn't exist
        if not os.path.exists(result_file):
            with open(result_file, "w") as f:
                json.dump({"status": "crash", "error": e.stderr}, f)

@app.command()
def tune(
    model_name: str = typer.Option(..., help="Model to tune"),
    dataset_path: str = typer.Option(..., help="Dataset path"),
    target_global_batch_size: int = typer.Option(32, help="Effective batch size to maintain"),
    num_gpus: int = typer.Option(1, help="Number of GPUs available"),
    strategy_type: str = typer.Option("heuristic", help="Search strategy: 'heuristic' or 'bayesian'")
):
    """
    Start the Minitune Auto-Tuner (MAT).
    """
    console.rule("[bold green]Minitune Auto-Tuner (MAT)[/bold green]")
    console.print(f"Targeting Global Batch Size: {target_global_batch_size}")
    
    # 1. Initialize Strategy
    if strategy_type.lower() == "bayesian":
        strategy = BayesianSearch(target_global_batch_size, num_gpus, n_calls=10)
        console.print("[bold blue]Using Bayesian Search Strategy[/bold blue]")
    elif strategy_type.lower() == "heuristic":
        strategy = HeuristicSearch(target_global_batch_size, num_gpus)
        console.print("[bold blue]Using Heuristic Search Strategy[/bold blue]")
    else:
        console.print(f"[bold red]Unknown strategy: {strategy_type}. Using Heuristic as default.[/bold red]")
        strategy = HeuristicSearch(target_global_batch_size, num_gpus)
    
    # Temp file for IPC
    tmp_result_file = "tmp_autotune_result.json"
    
    results_table = Table(title="Search History")
    results_table.add_column("Micro-Batch", style="cyan")
    results_table.add_column("Accum", style="magenta")
    results_table.add_column("Memory (MB)", style="yellow")
    results_table.add_column("MFU", style="green")
    results_table.add_column("Status", style="white")

    # 2. Search Loop
    while True:
        config = strategy.next_trial()
        if config is None:
            break
            
        console.print(f"\n[bold]Running Trial:[/bold] BS={config.micro_batch_size}, Accum={config.grad_accum_steps}...")
        
        # Clean previous result
        if os.path.exists(tmp_result_file):
            os.remove(tmp_result_file)
            
        # Launch Process
        run_subprocess_trial(
            model_name, 
            dataset_path, 
            config.micro_batch_size, 
            config.grad_accum_steps, 
            tmp_result_file
        )
        
        # Read Result
        if os.path.exists(tmp_result_file):
            with open(tmp_result_file, "r") as f:
                result_data = json.load(f)
        else:
            result_data = {"status": "crash"}

        # Feedback to Strategy
        strategy.report_result(config, result_data)
        
        # UI Update
        status = result_data.get("status", "unknown")
        mfu = result_data.get("mfu", 0.0)
        mem = result_data.get("peak_memory_mb", 0.0)
        
        color = "green" if status == "success" else "red"
        results_table.add_row(
            str(config.micro_batch_size),
            str(config.grad_accum_steps),
            f"{mem:.0f}",
            f"{mfu:.4f}",
            f"[{color}]{status}[/{color}]"
        )
        console.print(f"Result: {status.upper()} | MFU: {mfu:.4f}")

    # 3. Final Report
    console.print("\n")
    console.print(results_table)
    
    if strategy.best_config:
        console.rule("[bold gold1]Optimal Configuration Found[/bold gold1]")
        console.print(f"Micro-Batch Size: [bold green]{strategy.best_config.micro_batch_size}[/bold green]")
        console.print(f"Gradient Accumulation: [bold green]{strategy.best_config.grad_accum_steps}[/bold green]")
        console.print(f"Predicted MFU: [bold green]{strategy.best_mfu:.4f}[/bold green]")
        
        # Comparison with Baseline (Batch Size 1)
        baseline_mfu = strategy.history[0]["result"].get("mfu", 0.01)
        improvement = strategy.best_mfu / baseline_mfu
        console.print(f"Speedup vs Baseline: [bold yellow]{improvement:.2f}x[/bold yellow]")
    else:
        console.print("[bold red]No valid configuration found![/bold red]")

    # Cleanup
    if os.path.exists(tmp_result_file):
        os.remove(tmp_result_file)

if __name__ == "__main__":
    app()