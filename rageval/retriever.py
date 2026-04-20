"""FAISS-based retriever — loads a persisted index and returns relevant Documents."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rageval.architecture.base import BaseRetriever
from rageval.config import Config
from rageval.ingest import Document

logger = logging.getLogger(__name__)


class FAISSRetriever(BaseRetriever):
    """Retrieve the most relevant document chunks for a query using FAISS similarity search."""

    def __init__(self, config: Config) -> None:
        """Initialise the retriever by loading the FAISS index and embedding model."""
        self._config = config
        index_dir: Path = config.index_path

        index_file = index_dir / "index.faiss"
        docs_file = index_dir / "documents.json"

        if not index_file.exists() or not docs_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_dir}. "
                "Run 'rageval ingest' first to build the index."
            )

        logger.info("Loading FAISS index from %s", index_dir)
        self._index: faiss.IndexFlatL2 = faiss.read_index(str(index_file))

        raw_docs: list[dict] = json.loads(docs_file.read_text(encoding="utf-8"))
        self._documents: list[Document] = [
            Document(
                text=d["text"],
                source=d["source"],
                chunk_id=d["chunk_id"],
                metadata=d.get("metadata", {}),
            )
            for d in raw_docs
        ]

        logger.info("Index loaded: %d vectors, %d documents", self._index.ntotal, len(self._documents))

        if self._index.ntotal != len(self._documents):
            raise ValueError(
                f"Index/document mismatch: {self._index.ntotal} vectors vs "
                f"{len(self._documents)} documents. Re-run 'rageval ingest'."
            )

        logger.info("Loading embedding model: %s", config.embedding_model)
        self._model: SentenceTransformer = SentenceTransformer(config.embedding_model)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Embed a query and return the top-k most similar documents."""
        k = top_k if top_k is not None else self._config.top_k
        k = min(k, self._index.ntotal)  # don't request more than we have

        query_embedding: np.ndarray = self._model.encode(
            [query], convert_to_numpy=True
        ).astype(np.float32)

        distances, indices = self._index.search(query_embedding, k)  # type: ignore[arg-type]

        results: list[Document] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 for padding when k > ntotal
            doc = self._documents[idx]
            results.append(
                Document(
                    text=doc.text,
                    source=doc.source,
                    chunk_id=doc.chunk_id,
                    metadata={**doc.metadata, "distance": float(dist)},
                )
            )

        logger.info("Retrieved %d chunks for query: %.60s…", len(results), query)
        return results
