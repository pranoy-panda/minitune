# minitune/data.py
from datasets import load_dataset, Dataset, DatasetDict
from .config import DataConfig

def load_and_prepare_dataset(
    config: DataConfig,
    tokenizer,
    split: str = "train"
):
    """
    Universal data loader. 
    Handles 'chat' format (list of dicts) and 'instruction' format (columns).
    Returns (train_dataset, test_dataset) if split is None, else single Dataset.
    """
    print(f"Loading dataset: {config.path}...")
    
    # 1. Load the Dataset
    # Check if path is a local file
    if config.path.endswith((".json", ".jsonl")):
        # 'data_files' argument is required for local files
        raw_dataset = load_dataset("json", data_files=config.path, split=split)
    elif config.path.endswith(".parquet"):
        raw_dataset = load_dataset("parquet", data_files=config.path, split=split)
    elif config.path.endswith(".csv"):
        raw_dataset = load_dataset("csv", data_files=config.path, split=split)
    else:
        # Fallback to Hugging Face Hub
        raw_dataset = load_dataset(config.path, split=split)

    # 2. Handle Splitting (if a single split was loaded but we want train/test)
    # If the user passed split="train" but wants a validation set from it:
    if config.test_size > 0 and isinstance(raw_dataset, Dataset):
        print(f"Splitting dataset (Test Size: {config.test_size})...")
        # splits is a DatasetDict {'train': ..., 'test': ...}
        splits = raw_dataset.train_test_split(test_size=config.test_size)
    else:
        # Wrap single dataset in dict for uniform processing
        splits = DatasetDict({"train": raw_dataset})

    # 3. Define Formatting Logic
    def format_fn(example):
        """
        Converts raw example -> {"text": "formatted string"}
        """
        # Strategy A: Instruction Format (Prompt + Response columns)
        if config.format_type == "instruction":
            prompt = example[config.prompt_column]
            response = example[config.response_column] if config.response_column else ""
            
            # Construct standard message list
            messages = [{"role": "user", "content": prompt}]
            if response:
                messages.append({"role": "assistant", "content": response})
        
        # Strategy B: Chat Format (Already a list of dicts)
        else:
            messages = example[config.chat_column]

        # Apply Template
        # For SFT, we want the full text. 
        # For RL (generation), we might want add_generation_prompt=True (TODO)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    # 4. Apply Formatting (Map)
    # We use .map() to handle large datasets without loading everything into RAM
    print("Formatting and Tokenizing...")
    formatted_dataset = splits.map(format_fn)

    # 5. Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_input_token_length, 
            padding=False    # DataCollator will pad dynamically
        )

    # Map tokenization and remove raw text columns to save RAM
    tokenized_dataset = formatted_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=formatted_dataset["train"].column_names
    )

    # 6. Return
    # If we created a split, return tuple. If we just loaded one split, return it.
    if "test" in tokenized_dataset:
        return tokenized_dataset["train"], tokenized_dataset["test"]
    else:
        return tokenized_dataset["train"]