"""Adapter routing and retry logic for inference."""

from pathlib import Path
from typing import Callable, Optional, Union

from transformers import PreTrainedModel, PreTrainedTokenizer

from .generator import generate_response
from .validator import (
    validate_mcq_response,
    validate_saq_response,
    parse_mcq_response,
    parse_saq_response,
    get_format_error_message,
)
from ..prompts import build_prompt


def select_adapter(
    task_type: str,
    adapters_dir: Union[str, Path],
) -> Path:
    """Select the appropriate adapter path based on task type.

    Args:
        task_type: Either 'mcq' or 'saq'.
        adapters_dir: Base directory containing adapter subdirectories.

    Returns:
        Path to the adapter directory.

    Raises:
        ValueError: If task_type is not 'mcq' or 'saq'.
    """
    adapters_dir = Path(adapters_dir)

    if task_type == "mcq":
        return adapters_dir / "adapter_mcq"
    elif task_type == "saq":
        return adapters_dir / "adapter_saq"
    else:
        raise ValueError(f"Unknown task type: {task_type}. Must be 'mcq' or 'saq'.")


def generate_with_retry(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    task_type: str,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 0.0,
    use_stop_tokens: bool = False,
    stop_tokens: Optional[list] = None,
    max_retries: int = 2,
    validation_enabled: bool = True,
) -> str:
    """Generate a response with format validation and retry logic.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: Input prompt text.
        task_type: Either 'mcq' or 'saq'.
        max_new_tokens: Maximum tokens to generate.
        do_sample: Whether to use sampling.
        temperature: Sampling temperature.
        use_stop_tokens: Whether to use stop tokens.
        stop_tokens: List of stop token strings.
        max_retries: Maximum number of retry attempts.
        validation_enabled: Whether to validate and retry.

    Returns:
        Generated response text.
    """
    # Select validator and parser based on task type
    if task_type == "mcq":
        validator = validate_mcq_response
        parser = parse_mcq_response
    else:
        validator = validate_saq_response
        parser = parse_saq_response

    # First attempt
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        use_stop_tokens=use_stop_tokens,
        stop_tokens=stop_tokens,
    )

    if not validation_enabled:
        return parser(response)

    # Validate and retry if needed
    for attempt in range(max_retries):
        if validator(response):
            return parser(response)

        # Build retry prompt with error message
        error_msg = get_format_error_message(task_type)
        retry_context = f"Previous response: {response}\n\n{error_msg}"
        retry_prompt = build_prompt(tokenizer, error_msg, prompt.split("[/INST]")[-1] if "[/INST]" in prompt else prompt)

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=retry_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            use_stop_tokens=use_stop_tokens,
            stop_tokens=stop_tokens,
        )

    # Return parsed result even if validation fails after all retries
    return parser(response)
