"""Adapter routing and retry logic for inference."""

from datetime import datetime
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
    get_saq_format_requirement,
)
from ..prompts import build_prompt, MCQ_SYSTEM_PROMPT, SAQ_SYSTEM_PROMPT


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
    task_input: Optional[str] = None,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 0.0,
    use_stop_tokens: bool = False,
    stop_tokens: Optional[list] = None,
    max_retries: int = 2,
    validation_enabled: bool = True,
    log_retries: bool = False,
    retry_log_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate a response with format validation and retry logic.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: Input prompt text.
        task_type: Either 'mcq' or 'saq'.
        task_input: Raw task input (question/prompt) used to build retry prompts.
        max_new_tokens: Maximum tokens to generate.
        do_sample: Whether to use sampling.
        temperature: Sampling temperature.
        use_stop_tokens: Whether to use stop tokens.
        stop_tokens: List of stop token strings.
        max_retries: Maximum number of retry attempts.
        validation_enabled: Whether to validate and retry.
        log_retries: Whether to print retry debug logs.
        retry_log_path: Optional file path to append retry logs to.

    Returns:
        Generated response text.
    """
    retry_log_path = Path(retry_log_path) if retry_log_path else None

    def _truncate(s: str, n: int = 200) -> str:
        s = (s or "").replace("\n", "\\n")
        return s if len(s) <= n else (s[: n - 3] + "...")

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
        if task_type == "saq":
            if validator(response, question=task_input):
                return parser(response)
        else:
            if validator(response):
                return parser(response)

        if task_input is None:
            task_input = prompt

        # Build retry prompt with explicit formatting requirements
        error_msg = get_format_error_message(task_type)
        requirement = get_saq_format_requirement(task_input) if task_type == "saq" else None
        requirement_line = f"\n{requirement}\n" if requirement else "\n"

        if log_retries:
            ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            lines = [
                f"[{ts}] retry task={task_type} attempt={attempt + 1}/{max_retries}",
            ]
            if task_input:
                lines.append(f"[{ts}] input={_truncate(task_input, 160)}")
            lines.append(f"[{ts}] previous_response={_truncate(response)}")
            if requirement:
                lines.append(f"[{ts}] requirement={requirement}")

            if retry_log_path:
                retry_log_path.parent.mkdir(parents=True, exist_ok=True)
                with retry_log_path.open("a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
            else:
                for line in lines:
                    print(line, flush=True)

        if task_type == "saq":
            retry_user = (
                f"Question: {task_input}\n"
                f"Previous response: {response}\n\n"
                f"{error_msg}{requirement_line}"
                "Try again and follow the required format."
            )
            retry_prompt = build_prompt(tokenizer, SAQ_SYSTEM_PROMPT, retry_user)
        else:
            retry_user = (
                f"{task_input}\n"
                f"Previous response: {response}\n\n"
                f"{error_msg}\n"
                "Try again and follow the required format."
            )
            retry_prompt = build_prompt(tokenizer, MCQ_SYSTEM_PROMPT, retry_user)

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
