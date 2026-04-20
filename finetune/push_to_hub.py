"""Push LoRA adapter and model card to HuggingFace Hub."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

ADAPTER_DIR = Path(__file__).resolve().parent / "adapter"
RESULTS_FILE = Path(__file__).resolve().parent / "benchmark_results.json"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"


def _generate_model_card(username: str, repo_id: str) -> str:
    """Generate README.md for the model's Hub repository."""
    
    em_base, em_ft = 0.0, 0.0
    f1_base, f1_ft = 0.0, 0.0
    
    if RESULTS_FILE.exists():
        res = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        em_base = res.get("base", {}).get("exact_match_score", 0.0)
        em_ft = res.get("finetuned", {}).get("exact_match_score", 0.0)
        f1_base = res.get("base", {}).get("f1_score", 0.0)
        f1_ft = res.get("finetuned", {}).get("f1_score", 0.0)
        
    card = f"""---
base_model: {BASE_MODEL}
library_name: peft
tags:
- finance
- qlora
- lora
- phi3
---

# Phi-3-mini Finance Q&A LoRA

Phi-3-mini-4k-instruct fine-tuned with QLoRA on 320 synthetic finance Q&A pairs covering AR/AP workflows.

## Training Details
* **Dataset Size:** 320 pairs
* **Epochs:** 2
* **LoRA Rank:** 16
* **Target Modules:** q_proj, v_proj
* **Task:** Accounts Receivable / Accounts Payable Knowledge

## Benchmark Results
Evaluated on 80 held-out test pairs:

| Metric | Base Model | Fine-tuned |
|---|---|---|
| Exact Match | {em_base:.2f} | {em_ft:.2f} |
| F1 Score | {f1_base:.2f} | {f1_ft:.2f} |

## Intended Use and Limitations
Designed for answering specific FP&A, AP, and AR questions natively as part of an augmented workflow. Due to the small model size and synthetic data footprint, responses should be manually reviewed for absolute accuracy where financial compliance is necessary.

## How to use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    "{BASE_MODEL}", 
    load_in_4bit=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "{repo_id}")
```
"""
    return card


def push_to_hub() -> None:
    """Push the adapter and model card to HF Hub."""
    load_dotenv()
    
    hf_username = os.environ.get("HF_USERNAME")
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_username or not hf_token:
        console.print("[red]Error: HF_USERNAME and HF_TOKEN must be set in environment.[/red]")
        return
        
    repo_name = "phi3-mini-finance-qa-lora"
    repo_id = f"{hf_username}/{repo_name}"
    
    if not ADAPTER_DIR.exists():
        console.print(f"[red]Adapter directory not found at {ADAPTER_DIR}[/red]")
        return

    console.print(f"[bold cyan]Pushing adapter to {repo_id} on HuggingFace Hub[/bold cyan]")
    
    api = HfApi(token=hf_token)
    
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=hf_token)
        console.print(f"Created/verified repo: {repo_id}")
    except Exception as e:
        console.print(f"[red]Failed to create repo: {e}[/red]")
        return
        
    # Write model card locally to adapter dir before uploading
    readme_path = ADAPTER_DIR / "README.md"
    readme_path.write_text(_generate_model_card(hf_username, repo_id), encoding="utf-8")
    
    console.print("Uploading files...")
    api.upload_folder(
        folder_path=str(ADAPTER_DIR),
        repo_id=repo_id,
        repo_type="model",
    )
    
    console.print(f"[bold green]Successfully published to https://huggingface.co/{repo_id}[/bold green]")


if __name__ == "__main__":
    push_to_hub()
