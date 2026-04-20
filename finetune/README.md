# Fine-tuning Module

This directory contains the logic for synthetic data generation and LoRA fine-tuning.

## Workflow

1. **Synthetic Data Generation**: (`generate_data.py`) Uses a high-capability model (e.g., Claude 3.5 Sonnet or GPT-4o) to generate golden Q&A pairs from your raw documents.
2. **Training**: (`train.py`) Performs QLoRA fine-tuning on a smaller target model (e.g., Phi-3, Mistral) using the generated datasets.
3. **Evaluation & Benchmarking**: (`evaluate.py`) Compares the fine-tuned model's performance against the baseline.
4. **Export**: (`push_to_hub.py`) Pushes the trained adapter to HuggingFace Hub.

## Usage

Run the full flywheel from the root:
```bash
make all
```

Or individual steps:
```bash
make generate-data
make finetune
make benchmark
```

## Configuration

Fine-tuning parameters (learning rate, rank, alpha) are managed via `configs/config.yaml`.
