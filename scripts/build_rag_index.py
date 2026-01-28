#!/usr/bin/env python3
"""Build RAG index separately (before loading the model)."""

import argparse
import json
import gc
from pathlib import Path

from tqdm import tqdm
from rank_bm25 import BM25Okapi
import pickle


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization with lowercasing."""
    return text.lower().split()


def build_index(
    corpus_path: Path,
    index_dir: Path,
    max_articles: int = None,
    max_text_chars: int = 1500,
    k1: float = 1.5,
    b: float = 0.75,
):
    """Build BM25 index from corpus.

    Args:
        corpus_path: Path to JSONL corpus.
        index_dir: Directory to save index.
        max_articles: Limit number of articles (None = all).
        max_text_chars: Truncate article text to this many characters.
        k1: BM25 k1 parameter.
        b: BM25 b parameter.
    """
    corpus_path = Path(corpus_path)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Count lines for progress bar
    print("Counting articles...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    if max_articles:
        total_lines = min(total_lines, max_articles)

    print(f"Will process {total_lines:,} articles")
    print(f"Text truncated to {max_text_chars:,} chars per article")

    # Load and tokenize documents
    print("\nLoading and tokenizing documents...")
    documents = []
    tokenized_corpus = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Processing")):
            if max_articles and i >= max_articles:
                break

            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)
            title = doc.get("title", "")
            text = doc.get("text", "")[:max_text_chars]  # Truncate to save RAM
            documents.append({"title": title, "text": text})

            # Tokenize
            full_text = f"{title} {text}"
            tokenized_corpus.append(tokenize(full_text))

    print(f"\nLoaded {len(documents):,} documents")

    # Build BM25 index
    print("\nBuilding BM25 index (this may take a while)...")
    bm25_index = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    # Save documents
    print("\nSaving documents.json...")
    with open(index_dir / "documents.json", "w", encoding="utf-8") as f:
        json.dump(documents, f)

    # Save BM25 index
    print("Saving bm25_index.pkl...")
    with open(index_dir / "bm25_index.pkl", "wb") as f:
        pickle.dump(
            {
                "bm25_index": bm25_index,
                "tokenized_corpus": tokenized_corpus,
            },
            f,
        )

    # Save params for validation
    import hashlib
    with open(corpus_path, "rb") as f:
        corpus_hash = hashlib.md5(f.read(65536)).hexdigest()  # Hash first 64KB only

    with open(index_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump({"k1": k1, "b": b, "corpus_hash": corpus_hash}, f)

    # Print stats
    docs_size = (index_dir / "documents.json").stat().st_size / (1024 * 1024)
    index_size = (index_dir / "bm25_index.pkl").stat().st_size / (1024 * 1024)

    print(f"\nIndex built successfully!")
    print(f"  - documents.json: {docs_size:.1f} MB")
    print(f"  - bm25_index.pkl: {index_size:.1f} MB")
    print(f"  - Total: {docs_size + index_size:.1f} MB")
    print(f"\nSaved to: {index_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build RAG index from Wikipedia corpus")
    parser.add_argument(
        "--corpus", "-c",
        type=Path,
        required=True,
        help="Path to JSONL corpus file",
    )
    parser.add_argument(
        "--index-dir", "-o",
        type=Path,
        required=True,
        help="Output directory for index",
    )
    parser.add_argument(
        "--max-articles", "-n",
        type=int,
        default=None,
        help="Maximum articles to index (default: all)",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=1500,
        help="Truncate article text to N chars (default: 1800, saves RAM)",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter (default: 1.5)",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter (default: 0.75)",
    )

    args = parser.parse_args()

    build_index(
        corpus_path=args.corpus,
        index_dir=args.index_dir,
        max_articles=args.max_articles,
        max_text_chars=args.max_text_chars,
        k1=args.k1,
        b=args.b,
    )


if __name__ == "__main__":
    main()
