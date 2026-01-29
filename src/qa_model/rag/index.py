"""Wikipedia index building and loading for RAG."""

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi


class PorterStemmer:
    """Lightweight Porter Stemmer implementation."""

    def __init__(self):
        self.vowels = frozenset("aeiou")

    def _measure(self, stem: str) -> int:
        """Calculate the measure of a stem (number of VC patterns)."""
        cv = ""
        for char in stem:
            if char in self.vowels:
                cv += "V"
            else:
                cv += "C"
        return cv.count("VC")

    def _has_vowel(self, stem: str) -> bool:
        return any(c in self.vowels for c in stem)

    def _ends_double_consonant(self, word: str) -> bool:
        return len(word) >= 2 and word[-1] == word[-2] and word[-1] not in self.vowels

    def _ends_cvc(self, word: str) -> bool:
        if len(word) < 3:
            return False
        return (word[-3] not in self.vowels and
                word[-2] in self.vowels and
                word[-1] not in self.vowels and
                word[-1] not in "wxy")

    def stem(self, word: str) -> str:
        """Apply Porter stemming to a word."""
        if len(word) <= 2:
            return word

        # Step 1a
        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-2]
        elif word.endswith("ss"):
            pass
        elif word.endswith("s"):
            word = word[:-1]

        # Step 1b
        if word.endswith("eed"):
            if self._measure(word[:-3]) > 0:
                word = word[:-1]
        elif word.endswith("ed"):
            stem = word[:-2]
            if self._has_vowel(stem):
                word = stem
                if word.endswith(("at", "bl", "iz")):
                    word += "e"
                elif self._ends_double_consonant(word) and word[-1] not in "lsz":
                    word = word[:-1]
                elif self._measure(word) == 1 and self._ends_cvc(word):
                    word += "e"
        elif word.endswith("ing"):
            stem = word[:-3]
            if self._has_vowel(stem):
                word = stem
                if word.endswith(("at", "bl", "iz")):
                    word += "e"
                elif self._ends_double_consonant(word) and word[-1] not in "lsz":
                    word = word[:-1]
                elif self._measure(word) == 1 and self._ends_cvc(word):
                    word += "e"

        # Step 1c
        if word.endswith("y") and self._has_vowel(word[:-1]):
            word = word[:-1] + "i"

        # Step 2
        step2_suffixes = {
            "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
            "izer": "ize", "abli": "able", "alli": "al", "entli": "ent",
            "eli": "e", "ousli": "ous", "ization": "ize", "ation": "ate",
            "ator": "ate", "alism": "al", "iveness": "ive", "fulness": "ful",
            "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
        }
        for suffix, replacement in step2_suffixes.items():
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    word = stem + replacement
                break

        # Step 3
        step3_suffixes = {
            "icate": "ic", "ative": "", "alize": "al",
            "iciti": "ic", "ical": "ic", "ful": "", "ness": "",
        }
        for suffix, replacement in step3_suffixes.items():
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    word = stem + replacement
                break

        # Step 4
        step4_suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
            "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
            "ous", "ive", "ize",
        ]
        for suffix in step4_suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 1:
                    if suffix == "ion" and stem and stem[-1] in "st":
                        word = stem
                    elif suffix != "ion":
                        word = stem
                break

        # Step 5a
        if word.endswith("e"):
            stem = word[:-1]
            if self._measure(stem) > 1 or (self._measure(stem) == 1 and not self._ends_cvc(stem)):
                word = stem

        # Step 5b
        if self._measure(word) > 1 and self._ends_double_consonant(word) and word.endswith("l"):
            word = word[:-1]

        return word


# Global stemmer instance
_stemmer = PorterStemmer()


class WikipediaIndex:
    """In-memory Wikipedia index with BM25 retrieval."""

    # Common English stop words to filter out
    STOP_WORDS = frozenset([
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "also", "now",
        "and", "but", "if", "or", "because", "until", "while", "although",
        "this", "that", "these", "those", "what", "which", "who", "whom",
        "whose", "it", "its", "he", "she", "they", "them", "his", "her",
        "their", "i", "you", "we", "my", "your", "our", "me", "him", "us",
    ])

    def __init__(
        self,
        documents: List[Dict[str, str]],
        bm25_index: BM25Okapi,
        tokenized_corpus: List[List[str]],
    ):
        """Initialize the index.

        Args:
            documents: List of document dicts with 'title' and 'text' keys.
            bm25_index: Pre-built BM25Okapi index.
            tokenized_corpus: Tokenized corpus used for BM25.
        """
        self.documents = documents
        self.bm25_index = bm25_index
        self.tokenized_corpus = tokenized_corpus

    def __len__(self) -> int:
        return len(self.documents)

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """Tokenize text with lowercasing, stop word removal, and stemming."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        # Filter stop words, short tokens, and apply stemming
        return [_stemmer.stem(t) for t in tokens if t not in cls.STOP_WORDS and len(t) > 1]

    @classmethod
    def from_corpus(
        cls,
        corpus_path: Path,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "WikipediaIndex":
        """Build index from a JSONL corpus file.

        Args:
            corpus_path: Path to JSONL file with 'title' and 'text' fields.
            k1: BM25 k1 parameter.
            b: BM25 b parameter.

        Returns:
            WikipediaIndex instance.
        """
        documents = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    documents.append(doc)

        tokenized_corpus = [
            cls._tokenize(f"{doc.get('title', '')} {doc.get('text', '')}")
            for doc in documents
        ]

        bm25_index = BM25Okapi(tokenized_corpus, k1=k1, b=b)

        return cls(documents, bm25_index, tokenized_corpus)

    def save(self, index_dir: Path) -> None:
        """Save index to disk.

        Args:
            index_dir: Directory to save index files.
        """
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        # Save documents
        with open(index_dir / "documents.json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f)

        # Save BM25 index and tokenized corpus
        with open(index_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump(
                {
                    "bm25_index": self.bm25_index,
                    "tokenized_corpus": self.tokenized_corpus,
                },
                f,
            )

    @classmethod
    def load(cls, index_dir: Path) -> "WikipediaIndex":
        """Load index from disk.

        Args:
            index_dir: Directory containing index files.

        Returns:
            WikipediaIndex instance.
        """
        index_dir = Path(index_dir)

        with open(index_dir / "documents.json", "r", encoding="utf-8") as f:
            documents = json.load(f)

        with open(index_dir / "bm25_index.pkl", "rb") as f:
            data = pickle.load(f)

        return cls(documents, data["bm25_index"], data["tokenized_corpus"])

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, str], float]]:
        """Search for relevant documents.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of (document, score) tuples sorted by relevance.
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [(self.documents[i], scores[i]) for i in top_indices]


def build_wikipedia_index(
    corpus_path: Path,
    index_dir: Path,
    k1: float = 1.5,
    b: float = 0.75,
    force_rebuild: bool = False,
) -> WikipediaIndex:
    """Build or load Wikipedia index.

    Args:
        corpus_path: Path to JSONL corpus file.
        index_dir: Directory to store/load index.
        k1: BM25 k1 parameter.
        b: BM25 b parameter.
        force_rebuild: If True, rebuild even if index exists.

    Returns:
        WikipediaIndex instance.
    """
    corpus_path = Path(corpus_path)
    index_dir = Path(index_dir)

    # Check if index exists and is up-to-date
    documents_file = index_dir / "documents.json"
    bm25_file = index_dir / "bm25_index.pkl"
    params_file = index_dir / "params.json"

    index_exists = documents_file.exists() and bm25_file.exists()

    if index_exists and not force_rebuild:
        # Index exists - load it (skip hash validation for speed)
        return WikipediaIndex.load(index_dir)

    # Build new index
    index = WikipediaIndex.from_corpus(corpus_path, k1=k1, b=b)
    index.save(index_dir)

    # Save parameters for future validation
    corpus_hash = _compute_file_hash(corpus_path)
    with open(params_file, "w", encoding="utf-8") as f:
        json.dump({"k1": k1, "b": b, "corpus_hash": corpus_hash}, f)

    return index


def _compute_file_hash(file_path: Path, block_size: int = 65536) -> str:
    """Compute MD5 hash of the first block of a file.

    Args:
        file_path: Path to file.
        block_size: Size of first block to hash.

    Returns:
        Hex digest of file hash.
    """
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read(block_size)).hexdigest()
