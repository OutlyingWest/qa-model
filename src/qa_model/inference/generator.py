"""Text generation utilities."""

from typing import List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteria, StoppingCriteriaList


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for specific tokens."""

    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs
    ) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


@torch.inference_mode()
def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop_tokens: Optional[List[str]] = None,
) -> str:
    """Generate text continuation for a single prompt.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: Input prompt text.
        max_new_tokens: Maximum tokens to generate.
        do_sample: Whether to use sampling.
        temperature: Sampling temperature.
        top_p: Top-p (nucleus) sampling parameter.
        stop_tokens: Optional list of stop token strings.

    Returns:
        Generated text (without the prompt).
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Build stopping criteria if stop tokens provided
    stopping_criteria = None
    if stop_tokens:
        stop_token_ids = []
        for token in stop_tokens:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if token_ids:
                stop_token_ids.extend(token_ids)
        if stop_token_ids:
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    if stopping_criteria:
        generate_kwargs["stopping_criteria"] = stopping_criteria

    outputs = model.generate(**generate_kwargs)

    # Decode only the generated part
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    # Remove stop tokens from output if present
    if stop_tokens:
        for token in stop_tokens:
            if generated.endswith(token):
                generated = generated[:-len(token)]

    return generated.strip()


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 0.0,
    use_stop_tokens: bool = False,
    stop_tokens: Optional[List[str]] = None,
) -> str:
    """Generate a response with optional stop token handling.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: Input prompt text.
        max_new_tokens: Maximum tokens to generate.
        do_sample: Whether to use sampling.
        temperature: Sampling temperature.
        use_stop_tokens: Whether to use stop tokens.
        stop_tokens: List of stop token strings.

    Returns:
        Generated response text.
    """
    effective_stop_tokens = stop_tokens if use_stop_tokens else None

    return generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        stop_tokens=effective_stop_tokens,
    )
