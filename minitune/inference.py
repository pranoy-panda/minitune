from vllm import LLM, SamplingParams
from typing import List

def generate(
    model_path: str,
    prompts: List[str],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> List[str]:
    """
    Generates text completions for a list of prompts using VLLM.

    Args:
        model_path (str): Path to the fine-tuned model (e.g., SFT or PPO output).
        prompts (List[str]): A list of prompts to generate completions for.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.

    Returns:
        List[str]: A list of generated responses.
    """
    llm = LLM(model=model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # VLLM expects a chat template to be applied for generation prompts
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted_prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    outputs = llm.generate(formatted_prompts, sampling_params)
    
    responses = [output.outputs[0].text for output in outputs]
    return responses