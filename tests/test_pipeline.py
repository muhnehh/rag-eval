"""Tests for the rageval RAG pipeline components."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
try:
    from ragas.metrics.collections import faithfulness
except ImportError:
    from ragas.metrics import faithfulness

from rageval.config import Config
from rageval.ingest import Document, ingest
from rageval.pipeline import PipelineResult, RAGPipeline
from rageval.retriever import FAISSRetriever


@pytest.fixture
def mock_config(tmp_path: Path) -> Config:
    index_dir = tmp_path / "data" / "index"
    raw_dir = tmp_path / "data" / "raw"
    golden_dir = tmp_path / "data" / "golden"
    reports_dir = tmp_path / "reports"
    
    for d in [index_dir, raw_dir, golden_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    return Config(
        index_dir=str(index_dir),
        raw_dir=str(raw_dir),
        golden_path=str(golden_dir / "golden_qa.json"),
        reports_dir=str(reports_dir),
        chunk_size=128,
        chunk_overlap=16,
        top_k=2,
    )


def test_ingest_creates_index(mock_config: Config, tmp_path: Path):
    """Test that ingestion properly processes text and creates FAISS indexing artifacts."""
    raw_dir = Path(mock_config.raw_path)
    file1 = raw_dir / "doc1.md"
    file2 = raw_dir / "doc2.md"
    
    file1.write_text("Accounts receivable is the balance of money due to a firm.", encoding="utf-8")
    file2.write_text("DSO is days sales outstanding. It's an important metric.", encoding="utf-8")
    
    chunks_created = ingest(docs_dir=raw_dir, config=mock_config)
    assert chunks_created > 0
    
    index_dir = Path(mock_config.index_path)
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "documents.json").exists()


def test_retriever_returns_docs(mock_config: Config, tmp_path: Path):
    """Test that the retriever fetches Document instances of top_k."""
    # Run ingest first
    raw_dir = Path(mock_config.raw_path)
    (raw_dir / "doc.md").write_text("Accounts receivable overview text.", encoding="utf-8")
    ingest(docs_dir=raw_dir, config=mock_config)
    
    retriever = FAISSRetriever(mock_config)
    docs = retriever.retrieve("accounts receivable")
    
    assert len(docs) <= mock_config.top_k
    assert isinstance(docs[0], Document)
    assert "distance" in docs[0].metadata


@patch("litellm.completion")
def test_pipeline_answer_is_string(mock_completion, mock_config: Config):
    """Test RAG pipeline answer end-to-end format with mock LLM."""
    raw_dir = Path(mock_config.raw_path)
    (raw_dir / "doc.md").write_text("Accounts receivable overview text.", encoding="utf-8")
    ingest(docs_dir=raw_dir, config=mock_config)
    
    # Mock Litellm
    mock_msg = MagicMock()
    mock_msg.message.content = "Mocked answer string."
    mock_response = MagicMock()
    mock_response.choices = [mock_msg]
    mock_completion.return_value = mock_response

    pipeline = RAGPipeline(mock_config)
    res = pipeline.answer("What is AR?")
    
    assert isinstance(res, PipelineResult)
    assert res.answer == "Mocked answer string."
    assert res.latency_ms > 0
    assert len(res.retrieved_docs) > 0
