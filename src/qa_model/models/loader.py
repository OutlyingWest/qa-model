"""Model loading utilities with LoRA support."""

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from peft import PeftModel, get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel


@dataclass
class ModelBundle:
    """Container for model and tokenizer."""

    model_id: str
    tokenizer: AutoTokenizer
    model: PreTrainedModel


def load_base_model(
    model_id: str,
    cache_dir: Optional[str] = None,
    dtype: str = "float16",
    device_map: str = "auto",
) -> ModelBundle:
    """Load a base model and tokenizer.

    Args:
        model_id: HuggingFace model identifier.
        cache_dir: Directory to cache model files.
        dtype: Data type for model weights ('float16', 'bfloat16', 'float32').
        device_map: Device mapping strategy (default: 'auto').

    Returns:
        ModelBundle containing model and tokenizer.
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        use_fast=True,
    )

    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    return ModelBundle(model_id=model_id, tokenizer=tokenizer, model=model)


def apply_lora(model: PreTrainedModel, lora_config: LoraConfig) -> PeftModel:
    """Apply LoRA adapter to a model.

    Args:
        model: Base model to apply LoRA to.
        lora_config: LoRA configuration.

    Returns:
        PeftModel with LoRA adapter.
    """
    return get_peft_model(model, lora_config)


def load_adapter(
    model: PreTrainedModel,
    adapter_path: Union[str, Path],
    adapter_name: str = "default",
) -> PeftModel:
    """Load a trained LoRA adapter.

    Args:
        model: Base model to load adapter into.
        adapter_path: Path to saved adapter.
        adapter_name: Name for the adapter (default: 'default').

    Returns:
        PeftModel with loaded adapter.
    """
    return PeftModel.from_pretrained(
        model,
        adapter_path,
        adapter_name=adapter_name,
    )


def save_adapter(model: PeftModel, output_path: Union[str, Path]) -> None:
    """Save LoRA adapter weights.

    Args:
        model: PeftModel with adapter to save.
        output_path: Directory to save adapter to.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)


def unload_model(bundle: ModelBundle) -> None:
    """Free GPU/CPU memory by unloading model.

    Args:
        bundle: ModelBundle to unload.
    """
    try:
        del bundle.model
        del bundle.tokenizer
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
