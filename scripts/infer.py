#!/usr/bin/env python3
"""Inference script for LoRA-adapted models."""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qa_model.models import load_base_model, load_adapter, unload_model
from qa_model.inference import generate_with_retry, select_adapter
from qa_model.inference.router import predict_mcq_choice
from qa_model.inference.mcq_scorer import CountryPriorReranker
from qa_model.prompts import build_mcq_prompt, build_saq_prompt
from qa_model.rag import create_retriever, RAGRetriever


def run_mcq_inference(
    model,
    tokenizer,
    df: pd.DataFrame,
    cfg: DictConfig,
    retry_log_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run inference on MCQ dataset.

    Args:
        model: The model with adapter.
        tokenizer: The tokenizer.
        df: DataFrame with 'prompt' column.
        cfg: Inference configuration.

    Returns:
        DataFrame with predictions.
    """
    choices = []
    inference_cfg = cfg.inference

    mcq_mode = getattr(inference_cfg, "mcq_mode", "generate")
    mcq_logprob_variants = None
    if getattr(inference_cfg, "mcq_logprob", None) is not None and getattr(inference_cfg.mcq_logprob, "variants", None):
        mcq_logprob_variants = list(inference_cfg.mcq_logprob.variants)

    reranker = None
    rerank_weight = 0.0
    rerank_cfg = getattr(inference_cfg, "mcq_rerank", None)
    if rerank_cfg is not None and bool(getattr(rerank_cfg, "enabled", False)):
        rerank_weight = float(getattr(rerank_cfg, "weight", 0.0) or 0.0)
        alpha = float(getattr(rerank_cfg, "alpha", 1.0) or 1.0)
        train_csv = getattr(rerank_cfg, "train_csv", None) or getattr(rerank_cfg, "train_path", None)
        if train_csv is None:
            train_csv = str(Path(cfg.paths.data_dir) / "train_dataset_mcq.csv")
        train_csv_path = Path(train_csv)
        if rerank_weight and train_csv_path.exists():
            reranker = CountryPriorReranker.from_train_csv(train_csv_path, alpha=alpha)

    stop_tokens = list(inference_cfg.stop_tokens) if inference_cfg.stop_tokens else None
    top_p = float(getattr(inference_cfg, "top_p", 1.0) or 1.0)

    for row in tqdm(df.itertuples(index=False), desc="MCQ Inference"):
        prompt_text = getattr(row, "prompt")
        prompt = build_mcq_prompt(tokenizer, prompt_text)

        mcq_choices = None
        mcq_country = getattr(row, "country", None)
        if reranker is not None:
            try:
                mcq_choices = json.loads(getattr(row, "choices"))
            except Exception:
                mcq_choices = None

        choice = predict_mcq_choice(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            task_input=prompt_text,
            mcq_mode=mcq_mode,
            mcq_choices=mcq_choices,
            mcq_country=mcq_country,
            reranker=reranker,
            rerank_weight=rerank_weight,
            logprob_variants=mcq_logprob_variants,
            max_new_tokens=inference_cfg.max_new_tokens,
            do_sample=inference_cfg.do_sample,
            temperature=inference_cfg.temperature,
            top_p=top_p,
            use_stop_tokens=inference_cfg.use_stop_tokens,
            stop_tokens=stop_tokens,
            max_retries=inference_cfg.validation.max_retries,
            validation_enabled=inference_cfg.validation.enabled,
            log_retries=getattr(inference_cfg.validation, "log_retries", False),
            retry_log_path=retry_log_path,
        )
        choices.append(choice)

    # Build one-hot encoded submission
    result = pd.DataFrame({"MCQID": df["MCQID"]})
    for col in ["A", "B", "C", "D"]:
        result[col] = [c == col for c in choices]

    return result


def run_saq_inference(
    model,
    tokenizer,
    df: pd.DataFrame,
    cfg: DictConfig,
    retry_log_path: Optional[Path] = None,
    retriever: Optional[RAGRetriever] = None,
) -> pd.DataFrame:
    """Run inference on SAQ dataset.

    Args:
        model: The model with adapter.
        tokenizer: The tokenizer.
        df: DataFrame with 'en_question' column.
        cfg: Inference configuration.
        retry_log_path: Optional path for retry logs.
        retriever: Optional RAG retriever for context augmentation.

    Returns:
        DataFrame with predictions.
    """
    answers = []
    inference_cfg = cfg.inference

    # RAG settings
    rag_cfg = cfg.get("rag", {})
    rag_enabled = retriever is not None
    rag_top_k = int(rag_cfg.get("top_k", 3)) if rag_enabled else 0
    rag_max_tokens = int(rag_cfg.get("max_context_tokens", 512)) if rag_enabled else 0

    for question in tqdm(df["en_question"].tolist(), desc="SAQ Inference"):
        # Retrieve context if RAG is enabled
        context = None
        if rag_enabled:
            documents = retriever.retrieve(question, top_k=rag_top_k)
            if documents:
                context = retriever.format_context(documents, max_tokens=rag_max_tokens)

        prompt = build_saq_prompt(tokenizer, question, context=context)
        answer = generate_with_retry(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            task_type="saq",
            task_input=question,
            max_new_tokens=inference_cfg.max_new_tokens,
            do_sample=inference_cfg.do_sample,
            temperature=inference_cfg.temperature,
            top_p=float(getattr(inference_cfg, "top_p", 1.0) or 1.0),
            use_stop_tokens=inference_cfg.use_stop_tokens,
            stop_tokens=list(inference_cfg.stop_tokens) if inference_cfg.stop_tokens else None,
            max_retries=inference_cfg.validation.max_retries,
            validation_enabled=inference_cfg.validation.enabled,
            log_retries=getattr(inference_cfg.validation, "log_retries", False),
            retry_log_path=retry_log_path,
        )
        answers.append(answer)

    return pd.DataFrame({"ID": df["ID"], "answer": answers})


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main inference function.

    Args:
        cfg: Hydra configuration object.
    """
    print("=" * 80)
    print("LoRA Inference Script")
    print("=" * 80)
    print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Resolve paths
    data_dir = Path(cfg.paths.data_dir)
    adapters_dir = Path(cfg.paths.adapters_dir)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    retry_log_path = None
    if cfg.inference.validation.enabled and getattr(cfg.inference.validation, "log_retries", False):
        retry_log_file = getattr(cfg.inference.validation, "retry_log_file", "retry.log")
        retry_log_path = output_dir / retry_log_file

    task = cfg.task
    print(f"\nTask: {task}")
    print(f"Model: {cfg.model.model_id}")

    # Load base model
    print("\n[1/4] Loading base model...")
    bundle = load_base_model(
        model_id=cfg.model.model_id,
        cache_dir=cfg.model.cache_dir,
        dtype=cfg.model.dtype,
        device_map=cfg.model.device_map,
        max_memory=cfg.model.get("max_memory"),
    )
    print(f"Model loaded: {bundle.model_id}")

    # Load adapter (if enabled)
    print("\n[2/4] Loading adapter...")
    if not cfg.lora.enabled:
        print("LoRA adapters disabled (lora.enabled=false). Using base model.")
        model = bundle.model
    else:
        adapter_path = select_adapter(task, adapters_dir)
        if not adapter_path.exists():
            print(f"WARNING: Adapter not found at {adapter_path}. Running without adapter.")
            model = bundle.model
        else:
            model = load_adapter(bundle.model, adapter_path)
            print(f"Adapter loaded from: {adapter_path}")

    model.eval()

    # Initialize RAG retriever (only for SAQ)
    retriever = None
    rag_cfg = cfg.get("rag", {})
    if task == "saq" and rag_cfg.get("enabled", False):
        print("\n[3/5] Initializing RAG retriever...")
        try:
            retriever = create_retriever(rag_cfg)
            if retriever:
                print(f"RAG enabled: {rag_cfg.retriever.type} retriever initialized")
                print(f"  - Index: {rag_cfg.index.dir}")
                print(f"  - Top-k: {rag_cfg.top_k}")
                print(f"  - Max context tokens: {rag_cfg.max_context_tokens}")
        except FileNotFoundError as e:
            print(f"WARNING: RAG corpus/index not found ({e}). Running without RAG.")
            retriever = None
        except Exception as e:
            print(f"WARNING: Failed to initialize RAG ({e}). Running without RAG.")
            retriever = None
    else:
        if task == "saq":
            print("\n[3/5] RAG disabled (rag.enabled=false)")
        else:
            print("\n[3/5] RAG skipped (MCQ task)")

    # Load test data
    step = "[4/5]" if task == "saq" else "[3/4]"
    print(f"\n{step} Loading test data...")
    if task == "mcq":
        test_file = data_dir / "test_dataset_mcq.csv"
        df = pd.read_csv(test_file)
        print(f"Loaded {len(df)} MCQ test samples")
    else:
        test_file = data_dir / "test_dataset_saq.csv"
        df = pd.read_csv(test_file)
        print(f"Loaded {len(df)} SAQ test samples")

    # Run inference
    step = "[5/5]" if task == "saq" else "[4/4]"
    print(f"\n{step} Running inference...")
    if task == "mcq":
        result_df = run_mcq_inference(model, bundle.tokenizer, df, cfg, retry_log_path=retry_log_path)
    else:
        result_df = run_saq_inference(
            model, bundle.tokenizer, df, cfg,
            retry_log_path=retry_log_path,
            retriever=retriever,
        )

    # Save results
    output_file = output_dir / f"{task}_prediction.tsv"
    result_df.to_csv(output_file, sep="\t", index=False)

    # Cleanup
    print("\nCleaning up...")
    unload_model(bundle)

    print("\n" + "=" * 80)
    print("Inference complete!")
    print(f"Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
