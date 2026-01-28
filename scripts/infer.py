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
import json


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


def load_precomputed_contexts(contexts_file: Path) -> dict:
    """Load pre-computed RAG contexts from JSONL file.

    Args:
        contexts_file: Path to JSONL file with contexts.

    Returns:
        Dict mapping question ID to context string.
    """
    contexts = {}
    with open(contexts_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            contexts[record["id"]] = record.get("context", "")
    return contexts


def run_saq_inference(
    model,
    tokenizer,
    df: pd.DataFrame,
    cfg: DictConfig,
    retry_log_path: Optional[Path] = None,
    retriever: Optional[RAGRetriever] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 100,
    precomputed_contexts: Optional[dict] = None,
) -> pd.DataFrame:
    """Run inference on SAQ dataset with checkpointing support.

    Args:
        model: The model with adapter.
        tokenizer: The tokenizer.
        df: DataFrame with 'en_question' column.
        cfg: Inference configuration.
        retry_log_path: Optional path for retry logs.
        retriever: Optional RAG retriever for context augmentation.
        checkpoint_path: Path to save/load checkpoints.
        checkpoint_every: Save checkpoint every N questions.
        precomputed_contexts: Dict of pre-computed contexts (ID -> context).

    Returns:
        DataFrame with predictions.
    """
    inference_cfg = cfg.inference

    # RAG settings
    rag_cfg = cfg.get("rag", {})
    use_precomputed = precomputed_contexts is not None
    rag_enabled = retriever is not None or use_precomputed
    rag_top_k = int(rag_cfg.get("top_k", 3)) if rag_enabled else 0
    rag_max_tokens = int(rag_cfg.get("max_context_tokens", 512)) if rag_enabled else 0

    # Load checkpoint if exists
    answers = []
    start_idx = 0
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_df = pd.read_csv(checkpoint_path, sep="\t")
        answers = checkpoint_df["answer"].tolist()
        start_idx = len(answers)
        print(f"Resumed from checkpoint: {start_idx}/{len(df)} questions completed")

    questions = df["en_question"].tolist()
    ids = df["ID"].tolist()

    for idx in tqdm(range(start_idx, len(questions)), desc="SAQ Inference", initial=start_idx, total=len(questions)):
        question = questions[idx]
        question_id = ids[idx]

        # Get context: precomputed or retrieve on-the-fly
        context = None
        if use_precomputed:
            context = precomputed_contexts.get(question_id, "")
            if not context:
                context = None
        elif retriever is not None:
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

        # Save checkpoint
        if checkpoint_path and (idx + 1) % checkpoint_every == 0:
            checkpoint_df = pd.DataFrame({"ID": ids[:len(answers)], "answer": answers})
            checkpoint_df.to_csv(checkpoint_path, sep="\t", index=False)
            print(f"\nCheckpoint saved: {len(answers)}/{len(questions)}")

    # Final save
    if checkpoint_path:
        checkpoint_df = pd.DataFrame({"ID": ids[:len(answers)], "answer": answers})
        checkpoint_df.to_csv(checkpoint_path, sep="\t", index=False)

    return pd.DataFrame({"ID": ids, "answer": answers})


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

    # Initialize RAG (only for SAQ)
    retriever = None
    precomputed_contexts = None
    rag_cfg = cfg.get("rag", {})

    if task == "saq" and rag_cfg.get("enabled", False):
        print("\n[3/5] Initializing RAG...")

        # Check for pre-computed contexts first (faster)
        precomputed_path = rag_cfg.get("precomputed_contexts")
        if precomputed_path:
            precomputed_path = Path(precomputed_path)
            if precomputed_path.exists():
                print(f"Loading pre-computed contexts from {precomputed_path}...")
                precomputed_contexts = load_precomputed_contexts(precomputed_path)
                print(f"Loaded {len(precomputed_contexts)} pre-computed contexts")
            else:
                print(f"WARNING: Pre-computed contexts not found: {precomputed_path}")

        # Fall back to on-the-fly retrieval if no precomputed contexts
        if precomputed_contexts is None:
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

    # Checkpoint settings
    checkpoint_cfg = cfg.inference.get("checkpoint", {})
    checkpoint_enabled = bool(checkpoint_cfg.get("enabled", True))
    checkpoint_every = int(checkpoint_cfg.get("every", 100))
    checkpoint_path = output_dir / f"{task}_checkpoint.tsv" if checkpoint_enabled else None

    if checkpoint_enabled:
        print(f"Checkpointing enabled: saving every {checkpoint_every} questions to {checkpoint_path}")

    if task == "mcq":
        result_df = run_mcq_inference(model, bundle.tokenizer, df, cfg, retry_log_path=retry_log_path)
    else:
        result_df = run_saq_inference(
            model, bundle.tokenizer, df, cfg,
            retry_log_path=retry_log_path,
            retriever=retriever,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            precomputed_contexts=precomputed_contexts,
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
