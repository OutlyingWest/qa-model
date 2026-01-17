"""Response validation utilities for MCQ and SAQ tasks."""

import re
from typing import Optional, Tuple

# Regex patterns for parsing responses
_SAQ_ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_MCQ_CHOICE_RE = re.compile(r"\b([A-D])\b")

_HHMM_RE = re.compile(r"^\d{2}:\d{2}$")
_INT_RE = re.compile(r"^\d+$")
_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
_MONTH_1_12_RE = re.compile(r"^(?:1[0-2]|[1-9])$")
_RANGE_0_24_RE = re.compile(r"^(?:[0-9]|1[0-9]|2[0-4])$")
_RANGE_0_7_RE = re.compile(r"^(?:[0-7])$")


def _extract_saq_format_regex(question: str) -> Tuple[Optional[re.Pattern], Optional[str]]:
    """Infer an expected output regex from common dataset-style instructions."""
    q = (question or "").lower()

    if "hh:mm format" in q:
        return _HHMM_RE, r"^\d{2}:\d{2}$"

    if "1~12" in q:
        return _MONTH_1_12_RE, r"^(1[0-2]|[1-9])$"

    if "0~24" in q:
        return _RANGE_0_24_RE, r"^([0-9]|1[0-9]|2[0-4])$"

    if "0~7" in q:
        return _RANGE_0_7_RE, r"^[0-7]$"

    if "arabic numerals" in q:
        # Some questions explicitly allow a decimal point.
        if "decimal point" in q:
            return _NUMBER_RE, r"^\d+(\.\d+)?$"
        return _INT_RE, r"^\d+$"

    return None, None


def validate_mcq_response(text: str) -> bool:
    """Validate that a response contains a valid MCQ choice (A-D).

    Args:
        text: Response text to validate.

    Returns:
        True if valid MCQ response, False otherwise.
    """
    match = _MCQ_CHOICE_RE.search(text.upper())
    return match is not None


def validate_saq_response(text: str, question: Optional[str] = None) -> bool:
    """Validate that a response contains a valid SAQ answer format.

    Args:
        text: Response text to validate.
        question: Optional question text to enforce format constraints.

    Returns:
        True if valid SAQ response, False otherwise.
    """
    if not text or not text.strip():
        return False

    # Enforce single-line output (the prompt requires this).
    non_empty_lines = [line for line in (text or "").splitlines() if line.strip()]
    if len(non_empty_lines) != 1:
        return False

    # Require "Answer:" prefix to avoid accidental extra text getting parsed.
    if not _SAQ_ANSWER_RE.search(non_empty_lines[0]):
        return False

    parsed = parse_saq_response(non_empty_lines[0])
    if not parsed:
        return False

    # "idk" is always acceptable as an explicit unknown.
    if parsed == "idk":
        return True

    expected_re, _expected_re_str = _extract_saq_format_regex(question or "")
    if expected_re is None:
        return True

    return expected_re.fullmatch(parsed) is not None


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


def get_saq_format_requirement(question: str) -> Optional[str]:
    """Return a human-readable regex requirement string for retries, if any."""
    _expected_re, expected_re_str = _extract_saq_format_regex(question or "")
    if expected_re_str:
        return f"Output must match regex: {expected_re_str}"
    return None
