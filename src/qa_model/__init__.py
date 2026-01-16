"""QA Model - LoRA fine-tuning for MCQ and SAQ tasks."""

from . import data
from . import models
from . import training
from . import inference
from .prompts import (
    SAQ_SYSTEM_PROMPT,
    MCQ_SYSTEM_PROMPT,
    build_prompt,
    build_mcq_prompt,
    build_saq_prompt,
)

__version__ = "0.1.0"

__all__ = [
    "data",
    "models",
    "training",
    "inference",
    "SAQ_SYSTEM_PROMPT",
    "MCQ_SYSTEM_PROMPT",
    "build_prompt",
    "build_mcq_prompt",
    "build_saq_prompt",
]
