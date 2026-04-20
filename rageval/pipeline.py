"""RAG pipeline — query → retrieve context → LLM answer."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import litellm

from rageval.config import Config
from rageval.ingest import Document
from rageval.retriever import FAISSRetriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = (
    "You are a finance assistant. Answer ONLY using the provided context. "
    "If the context does not contain the answer, say "
    "'I don't know based on the provided context.' "
    "Do not use outside knowledge."
)


@dataclass
class PipelineResult:
    """Container for a single RAG pipeline response."""

    query: str
    answer: str
    retrieved_docs: list[Document] = field(default_factory=list)
    latency_ms: float = 0.0


class RAGPipeline:
    """Orchestrate retrieval-augmented generation: retrieve, format context, call LLM."""

    def __init__(self, config: Config) -> None:
        """Initialise pipeline components from config."""
        self._config = config
        self._retriever = FAISSRetriever(config)

    def _format_context(self, docs: list[Document]) -> str:
        """Concatenate retrieved document texts into a numbered context block."""
        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[{i}] (source: {doc.source}, chunk: {doc.chunk_id})\n{doc.text}")
        return "\n\n".join(parts)

    def answer(self, query: str) -> PipelineResult:
        """Run the full RAG pipeline for a single query and return a structured result."""
        start = time.perf_counter()

        # Retrieve
        docs = self._retriever.retrieve(query)
        context = self._format_context(docs)

        # Build messages
        user_message = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # LLM call
        try:
            response = litellm.completion(
                model=self._config.llm_model,
                messages=messages,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
            )
            answer_text: str = response.choices[0].message.content.strip()  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("LLM call failed for query '%.80s': %s", query, exc)
            raise RuntimeError(f"LLM completion failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Pipeline answered in %.1f ms: %.60s…", elapsed_ms, query)

        return PipelineResult(
            query=query,
            answer=answer_text,
            retrieved_docs=docs,
            latency_ms=elapsed_ms,
        )
