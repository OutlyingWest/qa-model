"""Response validation utilities for MCQ and SAQ tasks."""

import re
from typing import Optional, Tuple

# Regex patterns for parsing responses
_SAQ_ANSWER_RE = re.compile(r"\banswer\s*:\s*(.+)", re.IGNORECASE)
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
    """Extract a one-word SAQ answer from the model output.

    Args:
        text: Response text to parse.

    Returns:
        Extracted answer word, or 'idk' as fallback.
    """
    match = _SAQ_ANSWER_RE.search(text)
    if match:
        candidate = match.group(1).strip()
    else:
        candidate = text.strip()

    # Take the first token-ish segment; strip punctuation
    candidate = re.split(r"\s+|/|,|\bor\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
    candidate = candidate.strip().strip(".\"'`!?:;()[]{}<>")

    return candidate.lower() if candidate else "idk"


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
            "Please respond in the format: Answer: <your answer>. Explanation: <explanation>"
        )
