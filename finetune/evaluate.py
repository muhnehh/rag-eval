"""Benchmark exact match + F1 for base vs fine-tuned Phi-3 model."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)
console = Console()

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_DIR = "./finetune/adapter/"
TEST_DATA = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "finance_qa_test.json"
RESULTS_FILE = Path(__file__).resolve().parent / "benchmark_results.json"


@dataclass
class BenchmarkResult:
    """Results of a benchmark run for a model."""
    exact_match_score: float
    f1_score: float
    avg_latency_ms: float
    n_samples: int


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_base() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base Phi-3 model in 4-bit."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def load_finetuned() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model and apply the trained LoRA adapter."""
    base_model, tokenizer = load_base()
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    return model, tokenizer


def generate_answer(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, question: str) -> str:
    """Generate greedy response and extract the assistant's answer."""
    user_tag = "<|" + "user" + "|>"
    ast_tag = "<|" + "assistant" + "|>"
    
    prompt = f"{user_tag}\n{question}\n{ast_tag}\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,  # greedy
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant part
    parts = full_text.split(ast_tag.replace("<|", "").replace("|>", ""))
    if len(parts) > 1:
        return parts[-1].strip()
    return full_text.strip()


def exact_match(pred: str, gold: str) -> float:
    """Calculate exact match score (0 or 1)."""
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def token_f1(pred: str, gold: str) -> float:
    """Calculate token-level F1 score."""
    pred_tokens = pred.strip().lower().split()
    gold_tokens = gold.strip().lower().split()
    
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def run_benchmark(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, test_data: list[dict[str, str]]
) -> BenchmarkResult:
    """Run benchmark over test data and compute metrics."""
    em_scores = []
    f1_scores = []
    latencies = []

    for item in test_data:
        question = item["instruction"]
        gold = item["response"]

        start_time = time.perf_counter()
        pred = generate_answer(model, tokenizer, question)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        em_scores.append(exact_match(pred, gold))
        f1_scores.append(token_f1(pred, gold))
        latencies.append(elapsed_ms)

    n_samples = len(test_data)
    return BenchmarkResult(
        exact_match_score=sum(em_scores) / n_samples if n_samples > 0 else 0.0,
        f1_score=sum(f1_scores) / n_samples if n_samples > 0 else 0.0,
        avg_latency_ms=sum(latencies) / n_samples if n_samples > 0 else 0.0,
        n_samples=n_samples,
    )


def main() -> None:
    if not TEST_DATA.exists():
        console.print(f"[red]Test data not found at {TEST_DATA}[/red]")
        return
        
    test_data = json.loads(TEST_DATA.read_text(encoding="utf-8"))
    console.print(f"[cyan]Loaded {len(test_data)} test samples.[/cyan]")

    console.print("\n[bold]Evaluating Base Model[/bold]")
    base_model, base_tok = load_base()
    base_res = run_benchmark(base_model, base_tok, test_data)
    
    # Free memory
    del base_model
    torch.cuda.empty_cache()

    console.print("\n[bold]Evaluating Fine-tuned Model[/bold]")
    ft_model, ft_tok = load_finetuned()
    ft_res = run_benchmark(ft_model, ft_tok, test_data)

    # Format table
    table = Table(title="Benchmark Comparison")
    table.add_column("Metric", style="bold white")
    table.add_column("Base Model", justify="right")
    table.add_column("Fine-tuned", justify="right")
    table.add_column("Delta", justify="right")

    em_delta = ft_res.exact_match_score - base_res.exact_match_score
    f1_delta = ft_res.f1_score - base_res.f1_score
    
    em_color = "green" if em_delta >= 0 else "red"
    f1_color = "green" if f1_delta >= 0 else "red"
    
    table.add_row(
        "Exact Match",
        f"{base_res.exact_match_score:.2f}",
        f"{ft_res.exact_match_score:.2f}",
        f"[{em_color}]{em_delta:+.2f}[/{em_color}]"
    )
    table.add_row(
        "F1 Score",
        f"{base_res.f1_score:.2f}",
        f"{ft_res.f1_score:.2f}",
        f"[{f1_color}]{f1_delta:+.2f}[/{f1_color}]"
    )
    table.add_row(
        "Avg Latency (ms)",
        f"{base_res.avg_latency_ms:.1f}",
        f"{ft_res.avg_latency_ms:.1f}",
        f"{ft_res.avg_latency_ms - base_res.avg_latency_ms:+.1f}"
    )

    console.print("\n")
    console.print(table)

    results = {
        "base": asdict(base_res),
        "finetuned": asdict(ft_res),
        "delta": {
            "exact_match": em_delta,
            "f1": f1_delta,
            "latency": ft_res.avg_latency_ms - base_res.avg_latency_ms
        }
    }
    
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    console.print(f"\n[dim]Results saved to {RESULTS_FILE}[/dim]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
