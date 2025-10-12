import typer
from omegaconf import OmegaConf
from minitune.config import TrainConfig
from minitune.sft import SFTTrainer

def main(config_path: str):
    """
    Entry point for running SFT.
    """
    raw_config = OmegaConf.load(config_path)
    config = TrainConfig(**raw_config) # Validate with dataclass
    
    trainer = SFTTrainer(config)
    trainer.train()

if __name__ == "__main__":
    typer.run(main)