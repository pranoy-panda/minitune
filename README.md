# minitune

`minitune` is a small, research-focused Python library for post-training Large Language Models. It provides a clean and transparent codebase for common alignment tasks like Supervised Fine-Tuning (SFT) and Reinforcement Learning (DPO, GRPO), built on top of the Hugging Face ecosystem.

The primary goal of `minitune` is to be powerful enough for real research while being simple enough for learning and rapid experimentation. It avoids high-level abstractions like the Hugging Face `Trainer`.

In a nutshell, this library is currently my attempt to delve deeper into the knitty grity details of large scale model training, and share the same with the community.

### Core Principles

*   **Configuration over Code:** All experiment parameters are defined in simple YAML files.
*   **Transparency over Abstraction:** You have direct access to the PyTorch training loop, powered by `accelerate` for seamless distributed training and mixed-precision.
*   **Lean on the Ecosystem:** Uses the best-in-class libraries (`transformers`, `peft`, `vllm`, `accelerate`) for what they do best.

### Features

*   **Supervised Fine-Tuning (SFT):** Efficiently fine-tune models using PEFT (LoRA).
*   **Reinforcement Learning:**
*   **High-Performance Inference:** Integrated with `vllm` for fast and memory-efficient text generation.

### Installation

The project uses `uv` for fast and reliable package management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pranoy-panda/minitune.git
    cd minitune
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    # Create the virtual environment
    uv venv

    # Activate it (on Linux/macOS)
    source .venv/bin/activate

    # set cuda paths
    export CUDA_HOME="<replace with whatever ur path is, eg. /usr/local/cuda-12.4.0>"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

    # Install the project in editable mode with all dependencies
    uv pip install -e ".[dev]" --no-build-isolation

    # -no-build-isolation is required as to build `flash-attn` as it requires torch, but uv by default builds in isolation. So, this flag forces uv to build without isolation.
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


### Project Structure

```
minitune/
├── .gitignore
├── docs/
├── configs
│   ├── accelerate_fsdp_a40.yaml
│   └── minitune_sft_config_example.yaml
├── examples
│   ├── 01_run_sft.py
│   ├── 02_run_dpo.py
│   ├── __init__.py
├── minitune
│   ├── autotuner
│   │   ├── engine.py
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── strategies.py
│   │   └── trial.py
│   ├── config.py
│   ├── data.py
│   ├── inference.py
│   ├── __init__.py
│   ├── losses.py
│   ├── rl
│   │   ├── dpo.py
│   │   └── __init__.py
│   ├── sft.py
│   └── utils
│       └── profiling.py
├── LICENSE
└── pyproject.toml
```