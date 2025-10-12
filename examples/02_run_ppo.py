import typer
from omegaconf import OmegaConf
from minitune.config import TrainConfig
from minitune.rl.ppo import PPOTrainer

def main(config_path: str):
    """
    Entry point for running PPO.
    """
    raw_config = OmegaConf.load(config_path)
    config = TrainConfig(**raw_config)
    
    trainer = PPOTrainer(config)
    trainer.train()

if __name__ == "__main__":
    typer.run(main)