"""Abstract base classes defining the core rageval architecture layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol


class Document(Protocol):
    """Protocol for document chunks."""
    text: str
    source: str
    chunk_id: int


class BaseRetriever(ABC):
    """Abstract base class for retrieval logic."""

    @abstractmethod
    def retrieve(self, query: str) -> list[Document]:
        """Retrieve relevant documents for a query."""
        pass


class BaseEvaluator(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def evaluate(self, dataset: Any) -> dict[str, float]:
        """Compute metrics for a given dataset."""
        pass


class BasePipeline(ABC):
    """Abstract base class for RAG pipelines."""

    @abstractmethod
    def answer(self, query: str) -> Any:
        """Process a query and return an answer."""
        pass
