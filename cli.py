"""Command line interface for rageval."""

import time
from pathlib import Path

import click
from rich.console import Console

from rageval.config import load_config
from rageval.eval import run_eval, show_history
from rageval.ingest import ingest as _ingest
from rageval.pipeline import RAGPipeline

console = Console()


@click.group()
def cli():
    """rageval: CLI eval harness for RAG pipelines."""
    pass


@cli.command()
@click.option("--docs-dir", type=click.Path(exists=True), help="Path to raw documents directory")
@click.option("--config", type=click.Path(exists=True), help="Path to config.yaml")
def ingest(docs_dir, config):
    """Ingest documents: chunk, embed, and index into FAISS."""
    cfg = load_config(config)
    if docs_dir:
        cfg.raw_dir = docs_dir

    try:
        count = _ingest(docs_dir=docs_dir, config=cfg)
        console.print(f"[bold green]Successfully ingested {count} chunks into {cfg.index_dir}.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Ingestion failed: {e}[/bold red]")


@cli.command()
@click.option("--golden", type=click.Path(exists=True), help="Path to golden Q&A JSON")
@click.option("--config", type=click.Path(exists=True), help="Path to config.yaml")
def eval(golden, config):
    """Evaluate pipeline using RAGAS against golden dataset."""
    cfg = load_config(config)
    try:
        report = run_eval(golden_path=golden, config=cfg)
    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")


@cli.command()
@click.option("--query", "-q", required=True, help="Question to ask the RAG pipeline")
@click.option("--config", type=click.Path(exists=True), help="Path to config.yaml")
def ask(query, config):
    """Run a single query through the RAG pipeline."""
    cfg = load_config(config)
    try:
        pipeline = RAGPipeline(cfg)
        result = pipeline.answer(query)
        
        console.print(f"\n[bold green]Answer:[/bold green] {result.answer}")
        console.print(f"[bold blue]Latency:[/bold blue] {result.latency_ms:.1f}ms\n")
        
        console.print("[bold]Top 3 Sources:[/bold]")
        for i, doc in enumerate(result.retrieved_docs[:3]):
            console.print(f"  {i+1}. {doc.source} (chunk {doc.chunk_id})")
    except Exception as e:
        console.print(f"[bold red]Ask failed: {e}[/bold red]")


@cli.command()
@click.option("--config", type=click.Path(exists=True), help="Path to config.yaml")
def finetune(config):
    """Run fine-tuning workflow."""
    console.print("Fine-tune: see finetune/ directory or run `make finetune` directly to start QLoRA process.")


@cli.command()
@click.option("--config", type=click.Path(exists=True), help="Path to config.yaml")
def history(config):
    """Show evaluation history and score trends over time."""
    cfg = load_config(config)
    show_history(config=cfg)


if __name__ == "__main__":
    cli()
