from datasets import load_dataset, Dataset
from .config import DataConfig

def load_and_prepare_dataset(
    config: DataConfig,
    tokenizer,
    split: str = "train"
) -> Dataset:
    """
    Loads a dataset and prepares it for training by applying a chat template.
    For SFT, it uses prompt and response columns.
    For RL, it uses only the prompt column.
    """
    dataset = load_dataset(config.path, split=split)

    def format_sft(examples):
        prompts = examples[config.prompt_column]
        responses = examples[config.response_column]
        formatted_prompts = []
        for p, r in zip(prompts, responses):
            # This applies a chat template for a conversation turn
            messages = [
                {"role": "user", "content": p},
                {"role": "assistant", "content": r},
            ]
            formatted_prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ))
        return formatted_prompts

    def format_rl(examples):
        prompts = examples[config.prompt_column]
        formatted_prompts = []
        for p in prompts:
            messages = [
                {"role": "user", "content": p},
            ]
            # Here we add the generation prompt, signaling the model to respond
            formatted_prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        return formatted_prompts

    # SFT requires a response, RL does not. This determines the formatting.
    if config.response_column:
        texts = format_sft(dataset)
    else:
        texts = format_rl(dataset)

    # Tokenize the formatted text
    tokenized_dataset = tokenizer(
        texts, truncation=True, padding=False, max_length=1024
    )

    # Convert to a Hugging Face Dataset object
    # We need to structure it as a dictionary of lists
    data_dict = {
        key: list(values) for key, values in tokenized_dataset.items()
    }
    return Dataset.from_dict(data_dict)