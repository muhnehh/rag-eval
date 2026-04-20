"""QLoRA fine-tune of Phi-3-mini on synthetic finance Q&A data."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)
console = Console()

BASE_MODEL: str = "microsoft/Phi-3-mini-4k-instruct"
DATA_PATH: Path = (
    Path(__file__).resolve().parent.parent / "data" / "synthetic" / "finance_qa_train.json"
)
CHECKPOINT_DIR: str = "./finetune/checkpoints/"
ADAPTER_DIR: str = "./finetune/adapter/"


def _bnb_config() -> BitsAndBytesConfig:
    """Build a 4-bit NF4 BitsAndBytes quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def _load_base_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load tokenizer and 4-bit quantized base model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def _apply_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Wrap a base model with LoRA adapters and log trainable parameter stats."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    trainable, total = peft_model.get_nb_trainable_parameters()
    pct = 100.0 * trainable / total
    console.print(
        f"[cyan]Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)[/cyan]"
    )
    return peft_model


def _format_dataset(data_path: Path) -> Dataset:
    """Load JSON data and format it into HuggingFace Dataset for trl."""
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    
    # We assemble tags dynamically to avoid XML parser confusion
    user_tag = "<|" + "user" + "|>"
    ast_tag = "<|" + "assistant" + "|>"
    
    formatted = []
    for row in raw:
        text = f"{user_tag}\n{row['instruction']}\n{ast_tag}\n{row['response']}"
        formatted.append({"text": text})
    return Dataset.from_list(formatted)


def train() -> None:
    """Run QLoRA fine-tuning and save the adapter."""
    load_dotenv()
    console.print(f"[bold cyan]Starting QLoRA Fine-tuning[/bold cyan]")
    
    if not DATA_PATH.exists():
        console.print(f"[red]Training data not found at {DATA_PATH}[/red]")
        return

    console.print(f"Loading Base Model: {BASE_MODEL}")
    model, tokenizer = _load_base_model(BASE_MODEL)
    peft_model = _apply_lora(model)

    console.print(f"Preparing dataset from {DATA_PATH}")
    dataset = _format_dataset(DATA_PATH)

    training_args = SFTConfig(
        output_dir=CHECKPOINT_DIR,
        dataset_text_field="text",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        save_steps=50,
        logging_steps=10,
        max_seq_length=512,
        packing=False,
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
    )

    console.print("[bold green]Training...[/bold green]")
    start_time = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - start_time

    # Save adapter
    console.print(f"\n[bold cyan]Saving Adapter to {ADAPTER_DIR}[/bold cyan]")
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # Size calculation
    total_size = 0
    for path in Path(ADAPTER_DIR).rglob("*"):
        if path.is_file():
            total_size += path.stat().st_size
    size_mb = total_size / (1024 * 1024)

    # Logging results
    final_loss = train_result.metrics.get("train_loss", 0.0)
    console.print("[bold green]Training Complete![/bold green]")
    console.print(f"  Final Loss:    {final_loss:.4f}")
    console.print(f"  Total Time:    {elapsed:.1f} seconds")
    console.print(f"  Adapter Size:  {size_mb:.2f} MB")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
