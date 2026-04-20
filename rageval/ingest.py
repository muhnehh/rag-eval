"""Ingest documents — chunk, embed, and store in a FAISS index."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import faiss
import fitz  # PyMuPDF
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from rageval.config import Config, PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    """A single chunk of text with provenance metadata."""

    text: str
    source: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _extract_text_from_pdf(path: Path) -> str:
    """Extract all text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as exc:
        logger.error("Failed to extract text from PDF %s: %s", path, exc)
        raise IOError(f"Could not read PDF: {path}") from exc


def _extract_text_from_markdown(path: Path) -> str:
    """Read raw text from a Markdown or plain-text file."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to read text file %s: %s", path, exc)
        raise IOError(f"Could not read file: {path}") from exc


def _extract_text(path: Path) -> str:
    """Route a file to the correct text extractor based on its suffix."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_text_from_pdf(path)
    if suffix in {".md", ".markdown", ".txt", ".text"}:
        return _extract_text_from_markdown(path)
    raise ValueError(f"Unsupported file type: {suffix} (file: {path})")


SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".md", ".markdown", ".txt", ".text"}


def _discover_files(docs_dir: Path) -> list[Path]:
    """Recursively discover supported document files under a directory."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {docs_dir}")
    files = sorted(
        p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        logger.warning("No supported documents found in %s", docs_dir)
    return files


def _chunk_text(text: str, source: str, config: Config) -> list[Document]:
    """Split text into overlapping chunks and wrap each as a Document."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    raw_chunks: list[str] = splitter.split_text(text)

    documents: list[Document] = []
    for idx, chunk in enumerate(raw_chunks):
        chunk_id = hashlib.sha256(f"{source}::{idx}::{chunk[:64]}".encode()).hexdigest()[:12]
        documents.append(
            Document(
                text=chunk,
                source=source,
                chunk_id=chunk_id,
                metadata={"chunk_index": idx, "total_chunks": len(raw_chunks)},
            )
        )
    return documents


def _embed_documents(
    documents: list[Document], model: SentenceTransformer
) -> np.ndarray:
    """Embed a list of documents and return the embedding matrix."""
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    texts = [doc.text for doc in documents]
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=1, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def _save_index(
    index: faiss.IndexFlatL2,
    documents: list[Document],
    index_dir: Path,
) -> None:
    """Persist the FAISS index and document metadata to disk."""
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    meta = [asdict(doc) for doc in documents]
    (index_dir / "documents.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Index saved to %s (%d vectors)", index_dir, index.ntotal)


def ingest(docs_dir: str | Path | None = None, config: Config | None = None) -> int:
    """Ingest documents from a directory — chunk, embed, and build a FAISS index.

    Returns the total number of chunks indexed.
    """
    from dotenv import load_dotenv

    load_dotenv()

    if config is None:
        from rageval.config import load_config

        config = load_config()

    docs_path = Path(docs_dir) if docs_dir else config.raw_path
    index_dir = config.index_path

    logger.info("Discovering documents in %s", docs_path)
    files = _discover_files(docs_path)
    if not files:
        logger.warning("Nothing to ingest — no supported files found.")
        return 0

    # Chunk all files
    all_documents: list[Document] = []
    for fpath in files:
        logger.info("Processing %s", fpath.name)
        try:
            text = _extract_text(fpath)
            try:
                rel_source = str(fpath.relative_to(PROJECT_ROOT))
            except ValueError:
                rel_source = str(fpath.absolute())
            chunks = _chunk_text(text, rel_source, config)
            all_documents.extend(chunks)
            logger.info("  → %d chunks from %s", len(chunks), fpath.name)
        except Exception as exc:
            logger.error("Skipping %s due to error: %s", fpath.name, exc)

    if not all_documents:
        logger.warning("No chunks produced from any documents.")
        return 0

    # Embed
    logger.info("Loading embedding model: %s", config.embedding_model)
    model = SentenceTransformer(config.embedding_model)
    embeddings = _embed_documents(all_documents, model)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # type: ignore[arg-type]

    # Save
    _save_index(index, all_documents, index_dir)

    logger.info("Ingestion complete: %d documents, %d chunks", len(files), len(all_documents))
    return len(all_documents)
