import typer
from minitune.inference import generate
from rich import print

def main(
    model_path: str = typer.Option(..., help="Path to the fine-tuned model directory."),
    prompt: str = typer.Option("What is the meaning of life?", help="The prompt to send to the model.")
):
    """
    Runs inference using a fine-tuned model.
    """
    print(f"Loading model from: [bold cyan]{model_path}[/bold cyan]")
    print(f"Prompt: [bold yellow]'{prompt}'[/bold yellow]")

    responses = generate(model_path=model_path, prompts=[prompt])
    
    print("\n[bold green]Generated Response:[/bold green]")
    print(responses[0])

if __name__ == "__main__":
    typer.run(main)