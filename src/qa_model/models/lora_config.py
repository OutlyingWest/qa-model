"""LoRA configuration utilities."""

from typing import List, Optional

from peft import LoraConfig, TaskType


def create_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """Create a LoRA configuration.

    Args:
        r: LoRA rank (default: 16)
        alpha: LoRA alpha scaling factor (default: 32)
        dropout: LoRA dropout rate (default: 0.05)
        target_modules: List of module names to apply LoRA to.
            Defaults to attention projection layers.
        task_type: Task type for the model (default: CAUSAL_LM)

    Returns:
        LoraConfig object for use with PEFT.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=task_type,
        bias="none",
    )
