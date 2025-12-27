import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from rich.progress import Progress
from pathlib import Path
import typer
from omegaconf import OmegaConf

from minitune.config import TrainConfig

def train(config_path: str):
    # --- 1. SETUP ---
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    raw_config = OmegaConf.load(config_path)
    config = TrainConfig(**raw_config)

    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # --- 2. DATA LOADING (The Systems Way) ---
    
    # === OPTION A: The "Slow/OOM" Way (Commented Out) ===
    # Best for small datasets (<10GB). Loads everything into RAM.
    # requires DistributedSampler to split work.
    # ---------------------------------------------------------
    # raw_dataset = load_dataset(config.data.path, split="train")
    # # ... apply mapping/formatting here ...
    # sampler = DistributedSampler(raw_dataset, num_replicas=world_size, rank=global_rank)
    # shuffle_flag = False # Sampler handles shuffling
    # ---------------------------------------------------------

    # === OPTION B: The "Large Scale" Way (Active) ===
    # Best for TB-scale data. Uses Streaming. 
    # NO DistributedSampler needed; we shard the stream itself.
    # ---------------------------------------------------------
    raw_dataset = load_dataset(
        config.data.path, 
        split="train", 
        streaming=True # does not load full file to RAM
    )

    # Shard the data: Each rank only sees 1/Nth of the stream
    # e.g., Rank 0 sees examples 0, 4, 8... Rank 1 sees 1, 5, 9...
    sharded_dataset = raw_dataset.shard(num_shards=world_size, index=global_rank)

    # We must define the formatting logic inline for streaming
    def format_and_tokenize(example):
        text = f"User: {example[config.data.prompt_column]}\nAssistant: {example['response']}"
        return tokenizer(text, truncation=True, max_length=1024)

    # Map applies the function as data streams in
    train_dataset = sharded_dataset.map(format_and_tokenize, batched=False)
    
    # Streaming datasets don't support shuffle=True in DataLoader generally
    # We use a shuffle buffer instead
    train_dataset = train_dataset.shuffle(seed=42, buffer_size=10_000)

    sampler = None # We don't use a sampler with streaming
    shuffle_flag = False
    # ---------------------------------------------------------

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.sft.batch_size,
        collate_fn=data_collator,
        sampler=sampler, 
        shuffle=shuffle_flag,
        num_workers=2, 
        pin_memory=True
    )

    # --- 3. MODEL & GRADIENT CHECKPOINTING ---
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        use_flash_attention_2=config.model.use_flash_attention_2,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # === GRADIENT CHECKPOINTING SETUP ===
    # This reduces VRAM usage at the cost of some compute speed.
    # Essential for large models or long context.
    if hasattr(config, "sft") and getattr(config.sft, "gradient_checkpointing", False):
        print(f"Rank {global_rank}: Enabling Gradient Checkpointing")
        model.gradient_checkpointing_enable()
        
        # CRITICAL FOR PEFT/LORA: 
        # Checkpointing freezes the base model, so gradients stop flowing.
        # This re-enables input gradients so LoRA can update.
        model.enable_input_require_grads()

    # Apply PEFT
    if config.peft:
        peft_config = LoraConfig(**config.peft.__dict__)
        model = get_peft_model(model, peft_config)

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = AdamW(model.parameters(), lr=config.sft.learning_rate)

    # --- 4. TRAINING LOOP (Streaming Compatible) ---
    
    # With streaming, we often don't know __len__. 
    # We train for 'max_steps' instead of 'epochs'.
    max_steps = 1000 # Set this in config usually
    
    model.train()
    
    if global_rank == 0:
        print("Starting training...")

    step = 0
    # Loop over the stream indefinitely until max_steps
    for batch in train_dataloader:
        if step >= max_steps:
            break
            
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        step += 1
        if global_rank == 0 and step % config.sft.logging_steps == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

    # --- 5. CLEANUP ---
    if global_rank == 0:
        # Save logic (same as before)
        pass
    
    dist.destroy_process_group()

if __name__ == "__main__":
    typer.run(train)
