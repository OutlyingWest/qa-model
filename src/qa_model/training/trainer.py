"""Training utilities for LoRA fine-tuning."""

from pathlib import Path
from typing import Optional, Union, Callable, List

from datasets import Dataset
from peft import PeftModel
from transformers import TrainingArguments, PreTrainedTokenizer, TrainerCallback
from trl import SFTTrainer, SFTConfig


def create_training_args(
    output_dir: Union[str, Path],
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.1,
    scheduler: str = "cosine",
    logging_steps: int = 10,
    save_steps: int = 100,
    eval_steps: int = 100,
    save_total_limit: int = 2,
    fp16: bool = True,
    bf16: bool = False,
    mlflow_enabled: bool = False,
) -> SFTConfig:
    """Create training arguments for SFT.

    Args:
        output_dir: Directory to save checkpoints.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Learning rate.
        warmup_ratio: Warmup ratio for scheduler.
        scheduler: Learning rate scheduler type.
        logging_steps: Logging frequency.
        save_steps: Checkpoint save frequency.
        eval_steps: Evaluation frequency.
        save_total_limit: Maximum checkpoints to keep.
        fp16: Use FP16 mixed precision.
        bf16: Use BF16 mixed precision.
        mlflow_enabled: Enable MLflow tracking.

    Returns:
        SFTConfig object.
    """
    report_to = "mlflow" if mlflow_enabled else "none"
    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=scheduler,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=False,
        report_to=report_to,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )


def train_adapter(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    training_args: SFTConfig,
    callbacks: Optional[List[TrainerCallback]] = None,
) -> SFTTrainer:
    """Train a LoRA adapter using SFT.

    Args:
        model: PeftModel with LoRA adapter.
        tokenizer: Tokenizer for the model.
        train_dataset: Training dataset with 'text' column.
        val_dataset: Optional validation dataset.
        training_args: Training configuration (includes max_seq_length).
        callbacks: Optional list of trainer callbacks.

    Returns:
        Trained SFTTrainer object.
    """
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        callbacks=callbacks,
    )

    trainer.train()
    return trainer
