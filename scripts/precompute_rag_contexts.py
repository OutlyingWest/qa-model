#!/usr/bin/env python3
"""Pre-compute RAG contexts for all questions (runs without model loading)."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qa_model.rag import WikipediaIndex, BM25Retriever


def precompute_contexts(
    questions_file: Path,
    index_dir: Path,
    output_file: Path,
    top_k: int = 3,
    max_context_tokens: int = 512,
):
    """Pre-compute RAG contexts for all questions.

    Args:
        questions_file: CSV file with 'en_question' column.
        index_dir: Path to Wikipedia index.
        output_file: Output JSONL file with contexts.
        top_k: Number of documents to retrieve.
        max_context_tokens: Max context length.
    """
    print(f"Loading index from {index_dir}...")
    index = WikipediaIndex.load(index_dir)
    print(f"Loaded {len(index):,} documents")

    retriever = BM25Retriever(index=index, cache_enabled=False)

    print(f"\nLoading questions from {questions_file}...")
    df = pd.read_csv(questions_file)
    print(f"Loaded {len(df)} questions")

    print(f"\nPre-computing contexts (top_k={top_k}, max_tokens={max_context_tokens})...")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Retrieving"):
            question = row["en_question"]
            question_id = row["ID"]

            documents = retriever.retrieve(question, top_k=top_k)
            context = retriever.format_context(documents, max_tokens=max_context_tokens)

            record = {
                "id": question_id,
                "question": question,
                "context": context,
                "num_docs": len(documents),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved contexts to {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024*1024):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute RAG contexts")
    parser.add_argument(
        "--questions", "-q",
        type=Path,
        required=True,
        help="CSV file with questions (must have 'ID' and 'en_question' columns)",
    )
    parser.add_argument(
        "--index-dir", "-i",
        type=Path,
        required=True,
        help="Path to Wikipedia index directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output JSONL file for contexts",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Number of documents to retrieve (default: 3)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max context tokens (default: 512)",
    )

    args = parser.parse_args()

    precompute_contexts(
        questions_file=args.questions,
        index_dir=args.index_dir,
        output_file=args.output,
        top_k=args.top_k,
        max_context_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
