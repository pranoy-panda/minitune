# Advanced Supervised Fine-Tuning

This guide explains how to use `minitune` programmatically without relying solely on YAML configuration files. This is useful for:
1. Converting data on the fly (e.g., from Hugging Face).
2. Using custom loss functions (e.g., Focal Loss).

## 1. Programmatic Data Loading

Instead of letting the `SFTTrainer` load data from a file path defined in YAML, you can pass a pre-processed Hugging Face `Dataset` object directly.

```python
from datasets import load_dataset
from minitune.sft import SFTTrainer
from minitune.config import TrainConfig
from minitune.data import apply_chat_template

# 1. Load data from Hugging Face
raw_dataset = load_dataset("philschmid/dolly-15k-instruction-tuning-simplified", split="train")

# 2. Define a formatter function
def format_dolly(example):
    # Convert specific dataset columns to the standard chat format
    return {
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]}
        ]
    }

# 3. Apply standard formatting
# This function tokenizes and formats the data for the model
processed_dataset = apply_chat_template(
    dataset=raw_dataset,
    tokenizer_name="google/gemma-2b",
    formatting_func=format_dolly
)

# 4. Initialize Config (Minimal)
config = TrainConfig(model={"name_or_path": "google/gemma-2b"}, data={"path": ""})

# 5. Inject the dataset directly
trainer = SFTTrainer(config, train_dataset=processed_dataset)
trainer.train()
```

## 2. Custom Loss Functions

By default, `minitune` uses CrossEntropyLoss. You can inject custom logic to handle class imbalance or hard samples.

We support:
*   `FocalLoss`: Good for class imbalance.
*   `WeightedCrossEntropy`: For penalizing specific token errors.

```python
from minitune.losses import FocalLoss
from minitune.sft import SFTTrainer

# Initialize a custom loss
my_loss = FocalLoss(gamma=2.0, alpha=0.25)

# Pass it to the trainer
trainer = SFTTrainer(config, loss_fn=my_loss)
trainer.train()
```

