from dataclasses import dataclass, field
from typing import List, Optional

@dataclass #  a special type of class designed specifically to hold data
class ModelConfig:
    name_or_path: str
    use_flash_attention_2: bool = True

@dataclass
class PeftConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

@dataclass
class DataConfig:
    path: str
    split: str = "train"
    test_size: float = 0.1          # Auto-split validation percentage
    max_input_token_length: int = 2048 # Context Length
    
    # "chat" (list of dicts) or "instruction" (separate columns)
    format_type: str = "chat" 
    
    # Column mappings
    chat_column: str = "messages"       # For "chat" format
    prompt_column: str = "instruction"  # For "instruction" format
    response_column: str = "response"   # For "instruction" format

@dataclass
class SFTConfig:
    output_dir: str
    learning_rate: float = 2.0e-5
    epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    logging_steps: int = 10
    eval_steps: int = 50

@dataclass
class RLConfig:
    output_dir: str
    sft_model_path: str
    reward_model_path: str = "distilbert-base-uncased-finetuned-sst-2-english"
    learning_rate: float = 1.0e-6
    ppo_epochs: int = 4
    num_rollouts: int = 128
    batch_size: int = 4
    kl_penalty_coeff: float = 0.05
    max_new_tokens: int = 50

@dataclass
class DPOConfig:
    output_dir: str
    beta: float = 0.1          # The temperature/strength of the KL penalty
    learning_rate: float = 5.0e-7 # DPO usually needs lower LR than SFT
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    logging_steps: int = 10
    max_length: int = 1024
    max_prompt_length: int = 512
    
@dataclass
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    peft: Optional[PeftConfig] = None
    sft: Optional[SFTConfig] = None
    grpo: Optional[RLConfig] = None
    dpo: Optional[DPOConfig] = None