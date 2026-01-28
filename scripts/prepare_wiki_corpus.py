#!/usr/bin/env python3
"""Download and prepare Wikipedia corpus for RAG."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# Wikipedia dataset mapping (new Wikimedia format)
WIKI_DATASETS = {
    "en": "wikimedia/wikipedia",
    "simple": "wikimedia/wikipedia",
    "ru": "wikimedia/wikipedia",
    "de": "wikimedia/wikipedia",
    "fr": "wikimedia/wikipedia",
    "es": "wikimedia/wikipedia",
    "zh": "wikimedia/wikipedia",
    "ja": "wikimedia/wikipedia",
}


def prepare_wikipedia_corpus(
    output_path: Path,
    language: str = "en",
    max_articles: int = None,
    min_text_length: int = 100,
):
    """Download Wikipedia and save as JSONL corpus.

    Args:
        output_path: Path to output JSONL file.
        language: Wikipedia language code (e.g., 'en', 'ru', 'de').
        max_articles: Maximum number of articles to include (None = all).
        min_text_length: Minimum text length to include article.
    """
    print(f"Loading Wikipedia ({language})...")

    # Map language to config name
    if language == "simple":
        config_name = "20231101.simple"
    else:
        config_name = f"20231101.{language}"

    # Load Wikipedia dataset from Hugging Face (new format)
    dataset = load_dataset(
        "wikimedia/wikipedia",
        config_name,
        split="train",
        streaming=True,  # Use streaming to avoid downloading everything
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for article in tqdm(dataset, desc="Processing articles"):
            # Skip short articles
            text = article.get("text", "")
            if len(text) < min_text_length:
                continue

            doc = {
                "title": article.get("title", ""),
                "text": text,
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

            count += 1
            if max_articles and count >= max_articles:
                break

    print(f"\nSaved {count} articles to {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Prepare Wikipedia corpus for RAG")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/rag/wiki_corpus.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        help="Wikipedia language code (en, ru, de, fr, etc.) or 'simple' for Simple English",
    )
    parser.add_argument(
        "--max-articles", "-n",
        type=int,
        default=None,
        help="Maximum number of articles (default: all)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=100,
        help="Minimum text length to include article",
    )

    args = parser.parse_args()

    prepare_wikipedia_corpus(
        args.output,
        language=args.language,
        max_articles=args.max_articles,
        min_text_length=args.min_length,
    )


if __name__ == "__main__":
    main()
