from datasets import load_dataset, Dataset, DatasetDict
import torch
from torch.nn.utils.rnn import pad_sequence
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
    
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # 6. Return
    # If we created a split, return tuple. If we just loaded one split, return it.
    if "test" in tokenized_dataset:
        return tokenized_dataset["train"], tokenized_dataset["test"]
    else:
        return tokenized_dataset["train"]
    
    
# ------------ For DPO --------------
def prepare_dpo_data(config: DataConfig, tokenizer, split="train"):
    """
    Specialized loader for DPO.
    Returns a dataset with:
    - chosen_input_ids, chosen_attention_mask, chosen_labels
    - rejected_input_ids, rejected_attention_mask, rejected_labels
    """
    print(f" Loading DPO dataset: {config.path}...")
    dataset = load_dataset(config.path, split=split)
    # Assuming dataset has columns: prompt, chosen, rejected (lists of dicts or strings)
    
    # 1. Format Function
    def dpo_format_fn(example):
        # Extract raw strings
        if config.format_type == "preference":
            
            # Helper to get text from list-of-dicts or raw string
            def get_text(x):
                if isinstance(x, list): return tokenizer.apply_chat_template(x, tokenize=False)
                return x

            prompt = get_text(example["prompt"])
            chosen = get_text(example["chosen"])
            rejected = get_text(example["rejected"])
        else:
            raise ValueError(f"Unsupported format_type for DPO: {config.format_type}")

        # 2. Tokenize Parts
        # We need the prompt length to mask labels
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        chosen_tokens = tokenizer(chosen, add_special_tokens=False)["input_ids"] 
        rejected_tokens = tokenizer(rejected, add_special_tokens=False)["input_ids"]

        # 3. Build Full Sequences (Prompt + Response + EOS)
        # Note: We add tokenizer.bos_token_id at start if model requires it, 
        # usually handled by 'add_special_tokens=True' on the first chunk if we weren't splitting manually.
        # Here we manually build for precision.
        
        def build_seq(response_tokens):
            full_ids = prompt_tokens + response_tokens + [tokenizer.eos_token_id]
            # we need tell the loss function not to calculate loss on the prompt tokens, only on the response tokens.
            # thus as "-100" is the ignore_index in nn.CrossEntropyLoss by default, we set prompt tokens to -100 in labels.
            full_labels = [-100] * len(prompt_tokens) + response_tokens + [tokenizer.eos_token_id]
            return full_ids, full_labels

        chosen_ids, chosen_labels = build_seq(chosen_tokens)
        rejected_ids, rejected_labels = build_seq(rejected_tokens)

        # 4. Truncate (if beyond `max_len`)
        # We assume max_length covers most; rigorous impl needs careful truncation
        max_len = config.max_input_token_length
        return {
            "chosen_input_ids": chosen_ids[:max_len],
            "chosen_labels": chosen_labels[:max_len],
            "rejected_input_ids": rejected_ids[:max_len],
            "rejected_labels": rejected_labels[:max_len],
        }

    print(" Formatting & Tokenizing for DPO...")
    tokenized_dataset = dataset.map(dpo_format_fn, remove_columns=dataset.column_names)
    
    return tokenized_dataset

# Custom Collator for DPO (Pads dynamically)
def dpo_collate_fn(batch):
    batch_dict = {}
    for key in ["chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"]:
        # Pad with pad_token (or -100 for labels)
        padding_value = -100 if "labels" in key else 0 # 0 is usually safe for input_ids padding if masked
        
        tensors = [torch.tensor(x[key]) for x in batch]
        padded = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
        batch_dict[key] = padded
        
        # Create Attention Mask (1 for real tokens, 0 for pad)
        if "input_ids" in key:
            mask_key = key.replace("input_ids", "attention_mask")
            batch_dict[mask_key] = (padded != 0).long() # Assuming 0 is pad_id
            
    return batch_dict