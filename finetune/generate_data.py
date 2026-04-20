"""Generate synthetic finance Q&A pairs using the Anthropic API."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

SYSTEM_PROMPT: str = (
    "You are a finance expert specializing in accounts receivable (AR) and "
    "accounts payable (AP) processes. Generate diverse, realistic question-answer "
    "pairs that a finance AI assistant should know. Questions should cover: "
    "DSO calculation, invoice aging, three-way matching, payment terms (Net 30, "
    "Net 60), early payment discounts, credit limits, dunning workflows, PO matching, "
    "vendor reconciliation, GRNI (goods received not invoiced). Answers should be "
    "precise, 2-4 sentences, factually correct. Return a JSON array of "
    '{{"question", "answer"}} objects and nothing else.'
)

OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "synthetic"
PAIRS_PER_CALL: int = 20
NUM_CALLS: int = 20
TRAIN_RATIO: float = 0.8


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from an API response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _parse_pairs(raw_text: str) -> list[dict[str, str]]:
    """Parse a JSON array of question-answer pairs from raw response text."""
    cleaned = _strip_markdown_fences(raw_text)
    try:
        data = json.loads(cleaned)
        if not isinstance(data, list):
            logger.warning("Expected a JSON array, got %s", type(data).__name__)
            return []
        return [
            {"question": item["question"], "answer": item["answer"]}
            for item in data
            if isinstance(item, dict) and "question" in item and "answer" in item
        ]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error("Failed to parse Q&A pairs: %s", exc)
        return []


def _deduplicate(pairs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deduplicate pairs by normalised question text."""
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for p in pairs:
        key = p["question"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _to_hf_format(pairs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert raw pairs to HuggingFace instruction-response format."""
    return [
        {
            "instruction": p["question"],
            "response": p["answer"],
            "source": "synthetic-anthropic",
        }
        for p in pairs
    ]


def _save_json(data: list[dict[str, str]], path: Path) -> None:
    """Write a list of dicts to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %d records to %s", len(data), path)


def generate(
    num_calls: int = NUM_CALLS,
    pairs_per_call: int = PAIRS_PER_CALL,
    output_dir: Path | None = None,
) -> None:
    """Generate synthetic finance Q&A pairs via the Anthropic API and save to disk."""
    load_dotenv()
    import anthropic  # import here so non-finetune users don't need the dep

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Export it or add it to your .env file."
        )

    client = anthropic.Anthropic(api_key=api_key)
    out = output_dir or OUTPUT_DIR

    all_pairs: list[dict[str, str]] = []

    console.print(f"[bold cyan]Generating {num_calls * pairs_per_call} Q&A pairs…[/bold cyan]")

    for i in range(num_calls):
        console.print(f"  Batch {i + 1}/{num_calls}…", end=" ")
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Generate exactly {pairs_per_call} diverse finance Q&A pairs. "
                            "Cover different sub-topics each time. Return only the JSON array."
                        ),
                    }
                ],
            )
            raw_text = response.content[0].text  # type: ignore[union-attr]
            pairs = _parse_pairs(raw_text)
            all_pairs.extend(pairs)
            console.print(f"[green]{len(pairs)} pairs[/green]")
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
            logger.error("API call %d failed: %s", i + 1, exc)

    # Deduplicate
    before = len(all_pairs)
    all_pairs = _deduplicate(all_pairs)
    dups = before - len(all_pairs)

    # Split train / test
    split_idx = int(len(all_pairs) * TRAIN_RATIO)
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]

    # Convert to HF format
    train_hf = _to_hf_format(train_pairs)
    test_hf = _to_hf_format(test_pairs)
    full_hf = _to_hf_format(all_pairs)

    # Save
    _save_json(train_hf, out / "finance_qa_train.json")
    _save_json(test_hf, out / "finance_qa_test.json")
    _save_json(full_hf, out / "finance_qa_full.json")

    # Stats
    console.print("\n[bold cyan]Generation Statistics[/bold cyan]")
    console.print(f"  Total pairs generated:  {before}")
    console.print(f"  Duplicates removed:     {dups}")
    console.print(f"  Final unique pairs:     {len(all_pairs)}")
    console.print(f"  Train split:            {len(train_pairs)}")
    console.print(f"  Test split:             {len(test_pairs)}")
    console.print(f"\n  Saved to: {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate()
