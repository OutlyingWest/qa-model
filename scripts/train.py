#!/usr/bin/env python3
"""Training script for LoRA adapters."""

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qa_model.data import load_mcq_data, load_saq_data, split_dataset, MCQDataset, SAQDataset
from qa_model.models import load_base_model, apply_lora, save_adapter, unload_model
from qa_model.models.lora_config import create_lora_config
from qa_model.training import train_adapter, create_training_args
from qa_model.training.callbacks import (
    setup_mlflow,
    end_mlflow_run,
    log_params_to_mlflow,
    log_lora_params,
    MLflowSystemMetricsCallback,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object.
    """
    print("=" * 80)
    print("LoRA Training Script")
    print("=" * 80)
    print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Setup MLflow if enabled
    mlflow_cfg = cfg.get("mlflow", {})
    mlflow_enabled = mlflow_cfg.get("enabled", False)
    mlflow_run = None

    if mlflow_enabled:
        print("\nSetting up MLflow tracking...")
        mlflow_run = setup_mlflow(
            tracking_uri=mlflow_cfg.get("tracking_uri", "file:./mlruns"),
            experiment_name=mlflow_cfg.get("experiment_name", "qa-lora-finetuning"),
            run_name=f"{cfg.task}_{cfg.model.model_id.split('/')[-1]}",
            tags={"task": cfg.task, "model": cfg.model.model_id},
        )
        if mlflow_run:
            print(f"MLflow run started: {mlflow_run.info.run_id}")

    # Resolve paths
    data_dir = Path(cfg.paths.data_dir)
    adapters_dir = Path(cfg.paths.adapters_dir)

    # Determine task type from config
    task = cfg.task
    training_cfg = cfg.training

    print(f"\nTask: {task}")
    print(f"Model: {cfg.model.model_id}")

    # Load data based on task type
    print("\n[1/5] Loading data...")
    data_file = data_dir / training_cfg.data_file

    if task == "mcq":
        df = load_mcq_data(str(data_file))
        print(f"Loaded {len(df)} MCQ samples")
    else:
        df = load_saq_data(str(data_file))
        print(f"Loaded {len(df)} SAQ samples")

    # Split into train/val
    train_df, val_df = split_dataset(df, val_ratio=training_cfg.val_ratio)
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Load base model
    print("\n[2/5] Loading base model...")
    bundle = load_base_model(
        model_id=cfg.model.model_id,
        cache_dir=cfg.model.cache_dir,
        dtype=cfg.model.dtype,
        device_map=cfg.model.device_map,
    )
    print(f"Model loaded: {bundle.model_id}")

    # Create datasets
    print("\n[3/5] Preparing datasets...")
    if task == "mcq":
        train_dataset = MCQDataset(train_df, bundle.tokenizer, training_cfg.max_seq_length).to_hf_dataset()
        val_dataset = MCQDataset(val_df, bundle.tokenizer, training_cfg.max_seq_length).to_hf_dataset()
    else:
        train_dataset = SAQDataset(train_df, bundle.tokenizer, training_cfg.max_seq_length).to_hf_dataset()
        val_dataset = SAQDataset(val_df, bundle.tokenizer, training_cfg.max_seq_length).to_hf_dataset()

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Apply LoRA
    print("\n[4/5] Applying LoRA adapter...")
    lora_config = create_lora_config(
        r=cfg.lora.r,
        alpha=cfg.lora.alpha,
        dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
    )
    model_with_lora = apply_lora(bundle.model, lora_config)
    model_with_lora.print_trainable_parameters()

    # Log parameters to MLflow
    if mlflow_enabled and mlflow_run:
        # Log LoRA parameters
        lora_params = log_lora_params(lora_config)
        log_params_to_mlflow(lora_params)

        # Log training parameters
        log_params_to_mlflow({
            "model_id": cfg.model.model_id,
            "task_type": task,
            "batch_size": training_cfg.batch_size,
            "learning_rate": training_cfg.learning_rate,
            "epochs": training_cfg.epochs,
            "max_seq_length": training_cfg.max_seq_length,
            "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
            "warmup_ratio": training_cfg.warmup_ratio,
            "scheduler": training_cfg.scheduler,
        })

    # Create training arguments
    adapter_name = f"adapter_{task}"
    output_dir = adapters_dir / adapter_name

    training_args = create_training_args(
        output_dir=output_dir,
        epochs=training_cfg.epochs,
        batch_size=training_cfg.batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        learning_rate=training_cfg.learning_rate,
        warmup_ratio=training_cfg.warmup_ratio,
        scheduler=training_cfg.scheduler,
        logging_steps=training_cfg.logging_steps,
        save_steps=training_cfg.save_steps,
        eval_steps=training_cfg.eval_steps,
        save_total_limit=training_cfg.save_total_limit,
        fp16=training_cfg.fp16,
        bf16=training_cfg.bf16,
        mlflow_enabled=mlflow_enabled,
    )

    # Setup callbacks
    callbacks = []
    if mlflow_enabled and mlflow_cfg.get("log_system_metrics", True):
        callbacks.append(MLflowSystemMetricsCallback(
            log_every_n_steps=training_cfg.logging_steps
        ))

    # Train
    print("\n[5/5] Training...")
    trainer = train_adapter(
        model=model_with_lora,
        tokenizer=bundle.tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args,
        callbacks=callbacks if callbacks else None,
    )

    # Save final adapter
    print(f"\nSaving adapter to {output_dir}...")
    save_adapter(model_with_lora, output_dir)

    # Cleanup
    print("\nCleaning up...")
    unload_model(bundle)

    # End MLflow run
    if mlflow_enabled and mlflow_run:
        end_mlflow_run()
        print("MLflow run ended")

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Adapter saved to: {output_dir}")
    if mlflow_enabled:
        print(f"MLflow tracking: {mlflow_cfg.get('tracking_uri', 'file:./mlruns')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
