#!/usr/bin/env python3
"""RAG Pipeline Demo for SAQ.

Demonstrates how RAG (Retrieval-Augmented Generation) works for SAQ tasks:
1. Loading the Wikipedia index
2. Searching for relevant documents
3. Building prompts with context

Usage:
    python scripts/rag_demo.py --index-dir /path/to/wiki_index
    python scripts/rag_demo.py --index-dir /path/to/wiki_index --question "Who invented the telephone?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qa_model.rag import WikipediaIndex, BM25Retriever
from qa_model.prompts import build_saq_prompt, SAQ_SYSTEM_PROMPT, SAQ_RAG_SYSTEM_PROMPT


# Default test questions
DEFAULT_QUESTIONS = [
    "Who invented the telephone?",
    "What is the capital of France?",
    "When did World War II end?",
    "What is the chemical symbol for gold?",
    "Who wrote Romeo and Juliet?",
]


def demo_rag_pipeline(
    question: str,
    retriever: BM25Retriever,
    top_k: int = 3,
    max_tokens: int = 512,
) -> str:
    """Demonstrate RAG pipeline for a single question.

    Args:
        question: The question to process.
        retriever: BM25Retriever instance.
        top_k: Number of documents to retrieve.
        max_tokens: Max context tokens.

    Returns:
        The formatted context string.
    """
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    # 1. Search for relevant documents
    print(f"\n[1] BM25 search (top_k={top_k})...")
    documents = retriever.retrieve(question, top_k=top_k)

    print(f"\nFound {len(documents)} documents:")
    for i, doc in enumerate(documents, 1):
        title = doc.get("title", "N/A")
        text_preview = doc.get("text", "")[:350] + "..."
        print(f"\n  [{i}] {title}")
        print(f"      {text_preview}")

    # 2. Format context
    print(f"\n[2] Formatting context (max_tokens={max_tokens})...")
    context = retriever.format_context(documents, max_tokens=max_tokens)

    print(f"\nCONTEXT ({len(context)} chars):")
    print("-" * 40)
    print(context)
    print("-" * 40)

    return context


def show_full_prompt(question: str, context: str, tokenizer) -> str:
    """Show the full prompt with RAG context.

    Args:
        question: The question.
        context: The RAG context.
        tokenizer: The tokenizer for formatting.

    Returns:
        The formatted prompt with RAG context.
    """
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    # Prompt WITHOUT RAG
    prompt_no_rag = build_saq_prompt(tokenizer, question, context=None)

    # Prompt WITH RAG
    prompt_with_rag = build_saq_prompt(tokenizer, question, context=context)

    print("\n[WITHOUT RAG]")
    print("-" * 40)
    print(prompt_no_rag)
    print(f"\nTokens: {len(tokenizer.encode(prompt_no_rag))}")

    print("\n[WITH RAG]")
    print("-" * 40)
    print(prompt_with_rag)
    print(f"\nTokens: {len(tokenizer.encode(prompt_with_rag))}")

    return prompt_with_rag


def show_system_prompts():
    """Display the system prompts used for SAQ."""
    print("\n" + "=" * 80)
    print("SYSTEM PROMPTS COMPARISON")
    print("=" * 80)

    print("\nSAQ_SYSTEM_PROMPT (without RAG):")
    print("-" * 40)
    print(SAQ_SYSTEM_PROMPT)

    print("\n\nSAQ_RAG_SYSTEM_PROMPT (with RAG):")
    print("-" * 40)
    print(SAQ_RAG_SYSTEM_PROMPT)


def show_index_stats(index: WikipediaIndex, index_dir: Path):
    """Display index statistics.

    Args:
        index: The WikipediaIndex instance.
        index_dir: Path to the index directory.
    """
    print("\n" + "=" * 80)
    print("INDEX STATISTICS")
    print("=" * 80)

    print(f"  - Documents: {len(index):,}")

    # File sizes
    docs_file = index_dir / "documents.json"
    bm25_file = index_dir / "bm25_index.pkl"

    if docs_file.exists():
        size_mb = docs_file.stat().st_size / (1024 * 1024)
        print(f"  - documents.json: {size_mb:.1f} MB")

    if bm25_file.exists():
        size_mb = bm25_file.stat().st_size / (1024 * 1024)
        print(f"  - bm25_index.pkl: {size_mb:.1f} MB")

    # Sample document
    print("\nSample document from index:")
    sample_doc = index.documents[0]
    print(f"  Title: {sample_doc.get('title', 'N/A')}")
    print(f"  Text: {sample_doc.get('text', '')[:300]}...")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Demo for SAQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--index-dir", "-i",
        type=Path,
        required=True,
        help="Path to Wikipedia index directory",
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Custom question to test (optional, uses defaults if not provided)",
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
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model ID for tokenizer (default: meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--skip-prompts",
        action="store_true",
        help="Skip showing full prompts (faster, no tokenizer loading)",
    )

    args = parser.parse_args()

    # Load index
    print(f"Loading index from: {args.index_dir}")
    index = WikipediaIndex.load(args.index_dir)
    print(f"Loaded {len(index):,} documents")

    # Create retriever
    retriever = BM25Retriever(index=index, cache_enabled=False)

    # Determine questions to test
    questions = [args.question] if args.question else DEFAULT_QUESTIONS

    # Demo RAG pipeline for each question
    print("\n" + "=" * 80)
    print("RAG RETRIEVAL DEMO")
    print("=" * 80)

    contexts = {}
    for q in questions:
        contexts[q] = demo_rag_pipeline(
            q, retriever, top_k=args.top_k, max_tokens=args.max_tokens
        )
        print("\n")

    # Show full prompts (requires tokenizer)
    if not args.skip_prompts:
        print("\n" + "=" * 80)
        print("FULL PROMPTS WITH RAG CONTEXT")
        print("=" * 80)

        print(f"\nLoading tokenizer: {args.model_id}")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)

        for q in questions:
            show_full_prompt(q, contexts[q], tokenizer)
            print("\n\n")

        # Show system prompts comparison
        show_system_prompts()

    # Show index statistics
    show_index_stats(index, args.index_dir)

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
