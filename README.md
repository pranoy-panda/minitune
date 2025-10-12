# minitune

`minitune` is a small, research-focused Python library for post-training Large Language Models. It provides a clean and transparent codebase for common alignment tasks like Supervised Fine-Tuning (SFT) and Reinforcement Learning (PPO), built on top of the Hugging Face ecosystem.

The primary goal of `minitune` is to be powerful enough for real research while being simple enough for learning and rapid experimentation. It avoids high-level abstractions like the Hugging Face `Trainer` in favor of an explicit, `accelerate`-powered training loop that gives you full control.

### Core Principles

*   **Configuration over Code:** All experiment parameters are defined in simple YAML files.
*   **Transparency over Abstraction:** You have direct access to the PyTorch training loop, powered by `accelerate` for seamless distributed training and mixed-precision.
*   **Lean on the Ecosystem:** Uses the best-in-class libraries (`transformers`, `peft`, `vllm`, `accelerate`) for what they do best.
*   **Built for Research:** Easily extendable for custom loss functions, model architectures, and advanced training techniques.

### Features

*   **Supervised Fine-Tuning (SFT):** Efficiently fine-tune models using PEFT (LoRA).
*   **Reinforcement Learning (RLHF):** A clean implementation of Proximal Policy Optimization (PPO).
*   **High-Performance Inference:** Integrated with `vllm` for fast and memory-efficient text generation.

### Installation

The project uses `uv` for fast and reliable package management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/minitune.git
    cd minitune
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    # Create the virtual environment
    uv venv

    # Activate it (on Linux/macOS)
    source .venv/bin/activate

    # Install the project in editable mode with all dependencies
    uv pip install -e .
    ```

### Quickstart: Supervised Fine-Tuning (SFT)

1.  **Define a Configuration:**
    Create a YAML file (e.g., `configs/sft_example.yaml`) to define your experiment.

    ```yaml
    # configs/sft_example.yaml
    model:
      name_or_path: "google/gemma-2b"
      use_flash_attention_2: true

    peft:
      r: 16
      lora_alpha: 32
      target_modules:
        - "q_proj"
        - "v_proj"

    data:
      path: "examples/data/sft_data.jsonl"
      prompt_column: "prompt"
      response_column: "response"

    sft:
      output_dir: "./models/gemma-2b-sft"
      learning_rate: 2.0e-5
      epochs: 1
      batch_size: 1
      gradient_accumulation_steps: 4
      logging_steps: 1
    ```

2.  **Run the Training Script:**
    The `examples/01_run_sft.py` script shows how to use the `SFTTrainer`.

    ```bash
    # To run on a single GPU
    python examples/01_run_sft.py --config-path configs/sft_example.yaml

    # To run distributed training with accelerate
    # First, configure accelerate for your machine
    accelerate config
    # Then, launch the training run
    accelerate launch examples/01_run_sft.py --config-path configs/sft_example.yaml
    ```

### Quickstart: Reinforcement Learning (PPO)

1.  **Define a Configuration:**
    Create a config like `configs/ppo_example.yaml`. This assumes you have already run SFT and have a model checkpoint.

    ```yaml
    # configs/ppo_example.yaml
    model:
      name_or_path: "google/gemma-2b" # Base model for reference

    peft:
      r: 16 # Must match the SFT PEFT config
      lora_alpha: 32
      target_modules: ["q_proj", "v_proj"]

    data:
      path: "examples/data/rl_prompts.jsonl"
      prompt_column: "prompt"

    rl:
      output_dir: "./models/gemma-2b-ppo"
      sft_model_path: "./models/gemma-2b-sft" # <-- Path to your SFT model
      reward_model_path: "distilbert-base-uncased-finetuned-sst-2-english"
      learning_rate: 1.0e-6
      ppo_epochs: 4
      num_rollouts: 128
      batch_size: 4
      kl_penalty_coeff: 0.05
    ```

2.  **Run the Training Script:**
    Use the `examples/02_run_ppo.py` script.

    ```bash
    # Run with accelerate (recommended for PPO)
    accelerate launch examples/02_run_ppo.py --config-path configs/ppo_example.yaml
    ```

### Project Structure

```
minitune/
├── .gitignore
├── README.md
├── pyproject.toml
├── configs/
│   ├── ppo_example.yaml
│   └── sft_example.yaml
├── examples/
│   ├── __init__.py
│   ├── 01_run_sft.py
│   ├── 02_run_ppo.py
│   ├── 03_run_inference.py
│   └── data/
│       ├── rl_prompts.jsonl
│       └── sft_data.jsonl
└── minitune/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── inference.py
    ├── sft.py
    └── rl/
        ├── __init__.py
        ├── core.py
        └── ppo.py
```