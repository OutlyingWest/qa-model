"""RAG retriever implementations."""

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import DictConfig

from .index import WikipediaIndex, build_wikipedia_index


class RAGRetriever(ABC):
    """Abstract base class for RAG retrievers."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query.
            top_k: Number of documents to retrieve.

        Returns:
            List of document dicts with 'title' and 'text' keys.
        """
        pass

    @abstractmethod
    def format_context(
        self,
        documents: List[Dict[str, str]],
        max_tokens: int = 512,
    ) -> str:
        """Format retrieved documents as context string.

        Args:
            documents: List of retrieved documents.
            max_tokens: Maximum tokens for context (approximate).

        Returns:
            Formatted context string.
        """
        pass


class BM25Retriever(RAGRetriever):
    """BM25-based retriever using Wikipedia index."""

    def __init__(
        self,
        index: WikipediaIndex,
        cache_dir: Optional[Path] = None,
        cache_enabled: bool = True,
    ):
        """Initialize the BM25 retriever.

        Args:
            index: WikipediaIndex instance.
            cache_dir: Directory for query cache.
            cache_enabled: Whether to enable query caching.
        """
        self.index = index
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_enabled = cache_enabled and cache_dir is not None

        if self.cache_enabled and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "BM25Retriever":
        """Create retriever from Hydra configuration.

        Args:
            cfg: RAG configuration section.

        Returns:
            BM25Retriever instance.
        """
        # Build or load index
        corpus_path = Path(cfg.index.corpus)
        index_dir = Path(cfg.index.dir)

        bm25_cfg = cfg.retriever.get("bm25", {})
        k1 = float(bm25_cfg.get("k1", 1.5))
        b = float(bm25_cfg.get("b", 0.75))

        index = build_wikipedia_index(
            corpus_path=corpus_path,
            index_dir=index_dir,
            k1=k1,
            b=b,
        )

        # Setup cache
        cache_cfg = cfg.get("cache", {})
        cache_enabled = bool(cache_cfg.get("enabled", True))
        cache_dir = Path(cache_cfg.get("dir")) if cache_cfg.get("dir") else None

        return cls(
            index=index,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )

    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key for query."""
        key_str = f"{query}|{top_k}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[List[Dict[str, str]]]:
        """Load results from cache if available."""
        if not self.cache_enabled or not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_to_cache(self, cache_key: str, documents: List[Dict[str, str]]) -> None:
        """Save results to cache."""
        if not self.cache_enabled or not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(documents, f)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Retrieve relevant documents using BM25.

        Args:
            query: The search query.
            top_k: Number of documents to retrieve.

        Returns:
            List of document dicts with 'title' and 'text' keys.
        """
        # Check cache
        cache_key = self._get_cache_key(query, top_k)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        # Search index
        results = self.index.search(query, top_k=top_k)
        documents = [doc for doc, score in results if score > 0]

        # Cache results
        self._save_to_cache(cache_key, documents)

        return documents

    def format_context(
        self,
        documents: List[Dict[str, str]],
        max_tokens: int = 512,
    ) -> str:
        """Format retrieved documents as context string.

        Args:
            documents: List of retrieved documents.
            max_tokens: Maximum tokens for context (approximate, using 4 chars/token).

        Returns:
            Formatted context string.
        """
        if not documents:
            return ""

        # Approximate max chars (assuming ~4 chars per token)
        max_chars = max_tokens * 4

        context_parts = []
        current_chars = 0

        for doc in documents:
            title = doc.get("title", "")
            text = doc.get("text", "")

            # Format document
            doc_str = f"[{title}]\n{text}"

            # Check if adding this document would exceed limit
            if current_chars + len(doc_str) + 2 > max_chars:
                # Truncate text if needed
                remaining = max_chars - current_chars - len(title) - 5
                if remaining > 50:  # Only add if meaningful content remains
                    truncated_text = text[:remaining] + "..."
                    context_parts.append(f"[{title}]\n{truncated_text}")
                break

            context_parts.append(doc_str)
            current_chars += len(doc_str) + 2  # +2 for separator

        return "\n\n".join(context_parts)


def create_retriever(cfg: DictConfig) -> Optional[RAGRetriever]:
    """Factory function to create appropriate retriever.

    Args:
        cfg: RAG configuration section.

    Returns:
        RAGRetriever instance or None if RAG is disabled.
    """
    if not cfg.get("enabled", False):
        return None

    retriever_type = cfg.retriever.get("type", "bm25").lower()

    if retriever_type == "bm25":
        return BM25Retriever.from_config(cfg)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
