"""Pydantic settings loader — reads config from YAML and environment variables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Project root is two levels up from this file (rageval/rageval/config.py -> rageval/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default.yaml"


class Config(BaseModel):
    """Central configuration for the rageval pipeline."""

    # Ingestion
    chunk_size: int = Field(default=512, ge=64, le=8192, description="Token count per chunk")
    chunk_overlap: int = Field(default=64, ge=0, description="Overlap between consecutive chunks")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence-transformer model name")

    # Retrieval
    top_k: int = Field(default=5, ge=1, le=100, description="Number of chunks to retrieve")

    # LLM
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model identifier for litellm")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    llm_max_tokens: int = Field(default=512, ge=1, description="Max tokens in LLM response")

    # Paths (relative to project root)
    index_dir: str = Field(default="data/index", description="Directory for FAISS index files")
    raw_dir: str = Field(default="data/raw", description="Directory containing raw documents")
    golden_path: str = Field(default="data/golden/golden_qa.json", description="Path to golden Q&A set")
    reports_dir: str = Field(default="reports", description="Directory for eval report JSONs")

    # Eval
    eval_metrics: list[str] = Field(
        default=["faithfulness", "answer_relevancy", "context_recall"],
        description="RAGAS metrics to compute",
    )

    def resolve_path(self, relative: str) -> Path:
        """Resolve a config‐relative path against the project root."""
        return PROJECT_ROOT / relative

    @property
    def index_path(self) -> Path:
        """Absolute path to the FAISS index directory."""
        return self.resolve_path(self.index_dir)

    @property
    def raw_path(self) -> Path:
        """Absolute path to the raw documents directory."""
        return self.resolve_path(self.raw_dir)

    @property
    def golden_abs_path(self) -> Path:
        """Absolute path to the golden Q&A file."""
        return self.resolve_path(self.golden_path)

    @property
    def reports_path(self) -> Path:
        """Absolute path to the reports directory."""
        return self.resolve_path(self.reports_dir)


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from a YAML file, falling back to defaults."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning("Config file %s not found — using defaults.", path)
        return Config()

    try:
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        logger.info("Loaded config from %s", path)
        return Config(**raw)
    except yaml.YAMLError as exc:
        logger.error("Failed to parse YAML config at %s: %s", path, exc)
        raise ValueError(f"Invalid YAML in config file {path}") from exc
    except Exception as exc:
        logger.error("Unexpected error loading config from %s: %s", path, exc)
        raise
