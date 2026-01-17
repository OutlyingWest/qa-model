"""Response validation utilities for MCQ and SAQ tasks."""

import re
from typing import Optional, Tuple

# Regex patterns for parsing responses
_SAQ_ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_MCQ_CHOICE_RE = re.compile(r"\b([A-D])\b")


def validate_mcq_response(text: str) -> bool:
    """Validate that a response contains a valid MCQ choice (A-D).

    Args:
        text: Response text to validate.

    Returns:
        True if valid MCQ response, False otherwise.
    """
    match = _MCQ_CHOICE_RE.search(text.upper())
    return match is not None


def validate_saq_response(text: str) -> bool:
    """Validate that a response contains a valid SAQ answer format.

    Args:
        text: Response text to validate.

    Returns:
        True if valid SAQ response, False otherwise.
    """
    # Check for "Answer:" pattern
    match = _SAQ_ANSWER_RE.search(text)
    if match:
        answer = match.group(1).strip()
        # Ensure there's actually some content after "Answer:"
        return len(answer) > 0
    # Also accept direct answers without the "Answer:" prefix
    return len(text.strip()) > 0


def parse_mcq_response(text: str) -> str:
    """Extract a single choice letter (A-D) from the model output.

    Args:
        text: Response text to parse.

    Returns:
        Extracted choice letter, or 'A' as fallback.
    """
    match = _MCQ_CHOICE_RE.search(text.upper())
    if match:
        return match.group(1)
    # Deterministic fallback
    return "A"


def parse_saq_response(text: str) -> str:
    """Extract a short (possibly multiword) SAQ answer from the model output.

    Args:
        text: Response text to parse.

    Returns:
        Extracted answer string (1–6 tokens), or 'idk' as fallback.
    """
    match = _SAQ_ANSWER_RE.search(text or "")
    candidate = match.group(1) if match else (text or "")

    # Prefer the first line and drop any trailing sections if the model ignores the prompt.
    candidate = candidate.splitlines()[0]
    candidate = re.split(r"\bexplanation\s*:", candidate, maxsplit=1, flags=re.IGNORECASE)[0]

    # Normalize whitespace and strip outer punctuation.
    candidate = re.sub(r"\s+", " ", candidate.strip())
    candidate = candidate.strip().strip(".\"'`!?:;()[]{}<>")
    candidate = re.sub(r"\s+", " ", candidate).strip().lower()

    if not candidate:
        return "idk"

    # Enforce the multiword budget (matches the system prompt).
    tokens = candidate.split()
    if len(tokens) > 6:
        candidate = " ".join(tokens[:6])

    return candidate


def get_format_error_message(task_type: str) -> str:
    """Get a format error message for retry prompts.

    Args:
        task_type: Either 'mcq' or 'saq'.

    Returns:
        Error message string.
    """
    if task_type == "mcq":
        return (
            "Your previous response did not contain a valid choice. "
            "Please respond with exactly one letter: A, B, C, or D."
        )
    else:
        return (
            "Your previous response did not follow the required format. "
            "Please respond in exactly one line: Answer: <your answer>"
        )
