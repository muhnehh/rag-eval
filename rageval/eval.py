"""RAGAS evaluation runner — compute faithfulness, relevancy, and recall metrics."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness
from rich.console import Console
from rich.table import Table

from rageval.config import Config, load_config
from rageval.pipeline import RAGPipeline

logger = logging.getLogger(__name__)
console = Console()

METRIC_MAP = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_recall": context_recall,
}


@dataclass
class EvalReport:
    """Structured evaluation report with per-question details."""

    timestamp: str
    git_tag: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    per_question_results: list[dict[str, Any]] = field(default_factory=list)


def _get_git_tag() -> str | None:
    """Retrieve the current git tag, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("No git tag found or git not available.")
        return None


def _color_for_score(score: float) -> str:
    """Return a rich color name based on score thresholds."""
    if score >= 0.8:
        return "green"
    if score >= 0.6:
        return "yellow"
    return "red"


def _bar_for_score(score: float, width: int = 20) -> str:
    """Generate a colored text bar representing the score."""
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def print_metrics_table(metrics: dict[str, float]) -> None:
    """Display evaluation metrics as a rich table with colored score bars."""
    table = Table(title="RAG Evaluation Results", title_style="bold cyan")
    table.add_column("Metric", style="bold white", min_width=20)
    table.add_column("Score", justify="right", min_width=8)
    table.add_column("", min_width=22)

    for name, score in sorted(metrics.items()):
        color = _color_for_score(score)
        bar = _bar_for_score(score)
        table.add_row(
            name.replace("_", " ").title(),
            f"[{color}]{score:.2f}[/{color}]",
            f"[{color}]{bar}[/{color}]",
        )

    console.print(table)


def _create_git_tag(report: EvalReport) -> None:
    """Create a git tag annotated with the eval metrics."""
    tag_name = f"v{report.timestamp.replace(':', '-').replace(' ', '_')}"
    metrics_str = " ".join(f"{k}={v:.2f}" for k, v in sorted(report.metrics.items()))
    message = f"eval: {metrics_str}"

    try:
        subprocess.run(
            ["git", "tag", tag_name, "-m", message],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("Created git tag: %s", tag_name)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("Could not create git tag: %s", exc)


def run_eval(golden_path: str | Path | None = None, config: Config | None = None) -> EvalReport:
    """Run RAGAS evaluation against a golden Q&A set and produce a versioned report."""
    from dotenv import load_dotenv

    load_dotenv()

    if config is None:
        config = load_config()

    golden_file = Path(golden_path) if golden_path else config.golden_abs_path

    if not golden_file.exists():
        raise FileNotFoundError(
            f"Golden Q&A file not found at {golden_file}. "
            "Create it at data/golden/golden_qa.json."
        )

    # Load golden Q&A
    raw: list[dict[str, str]] = json.loads(golden_file.read_text(encoding="utf-8"))
    logger.info("Loaded %d golden Q&A pairs from %s", len(raw), golden_file)

    # Run pipeline on each question
    pipeline = RAGPipeline(config)

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []
    per_question: list[dict[str, Any]] = []

    for item in raw:
        question = item["question"]
        ground_truth = item["ground_truth"]

        console.print(f"  Evaluating: [dim]{question[:80]}[/dim]")
        result = pipeline.answer(question)

        questions.append(question)
        answers.append(result.answer)
        contexts.append([doc.text for doc in result.retrieved_docs])
        ground_truths.append(ground_truth)

        per_question.append({
            "question": question,
            "answer": result.answer,
            "ground_truth": ground_truth,
            "latency_ms": result.latency_ms,
            "num_chunks_retrieved": len(result.retrieved_docs),
        })

    # Build HuggingFace Dataset in RAGAS expected format
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Run RAGAS evaluation
    metrics_to_run = [METRIC_MAP[m] for m in config.eval_metrics if m in METRIC_MAP]
    logger.info("Running RAGAS with metrics: %s", [m.name for m in metrics_to_run])

    ragas_result = ragas_evaluate(eval_dataset, metrics=metrics_to_run)
    metrics: dict[str, float] = {k: float(v) for k, v in ragas_result.items() if isinstance(v, (int, float))}

    # Build report
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    git_tag = _get_git_tag()

    report = EvalReport(
        timestamp=timestamp,
        git_tag=git_tag,
        metrics=metrics,
        per_question_results=per_question,
    )

    # Save report
    reports_dir = config.reports_path
    reports_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{git_tag}" if git_tag else ""
    report_path = reports_dir / f"eval_{timestamp}{tag_suffix}.json"
    report_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Report saved to %s", report_path)

    # Create git tag for this eval run
    _create_git_tag(report)

    # Print results
    print_metrics_table(metrics)
    console.print(f"\n[dim]Report saved to {report_path}[/dim]")

    return report


def show_history(config: Config | None = None) -> None:
    """Read all eval reports and print a trend table sorted by timestamp."""
    if config is None:
        config = load_config()

    reports_dir = config.reports_path
    if not reports_dir.exists():
        console.print("[yellow]No reports directory found.[/yellow]")
        return

    report_files = sorted(reports_dir.glob("eval_*.json"))
    if not report_files:
        console.print("[yellow]No eval reports found in reports/.[/yellow]")
        return

    reports: list[EvalReport] = []
    for f in report_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            reports.append(EvalReport(**data))
        except Exception as exc:
            logger.warning("Skipping malformed report %s: %s", f.name, exc)

    reports.sort(key=lambda r: r.timestamp)

    # Build table
    table = Table(title="Evaluation History", title_style="bold cyan")
    table.add_column("Timestamp", style="dim")
    table.add_column("Git Tag", style="dim")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Relevancy", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Trend", justify="center")

    prev_avg: float | None = None
    for report in reports:
        faith = report.metrics.get("faithfulness", 0.0)
        relev = report.metrics.get("answer_relevancy", 0.0)
        recall = report.metrics.get("context_recall", 0.0)
        avg = (faith + relev + recall) / 3.0

        trend = ""
        if prev_avg is not None:
            if avg > prev_avg + 0.01:
                trend = "[green]▲[/green]"
            elif avg < prev_avg - 0.01:
                trend = "[red]▼[/red]"
            else:
                trend = "[dim]→[/dim]"
        prev_avg = avg

        table.add_row(
            report.timestamp,
            report.git_tag or "—",
            f"[{_color_for_score(faith)}]{faith:.2f}[/{_color_for_score(faith)}]",
            f"[{_color_for_score(relev)}]{relev:.2f}[/{_color_for_score(relev)}]",
            f"[{_color_for_score(recall)}]{recall:.2f}[/{_color_for_score(recall)}]",
            trend,
        )

    console.print(table)
