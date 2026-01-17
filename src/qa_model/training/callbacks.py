"""MLflow callbacks and metrics for training."""

import re
from typing import Dict, Any, Optional, List

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl, EvalPrediction
from transformers.training_args import TrainingArguments


def compute_mcq_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute metrics for MCQ task.

    Args:
        eval_pred: Evaluation predictions from trainer.

    Returns:
        Dictionary with accuracy and format_accuracy metrics.
    """
    predictions, labels = eval_pred

    # For language models, predictions are token ids
    # We need to decode and compare
    if hasattr(predictions, 'predictions'):
        predictions = predictions.predictions

    # Simple accuracy: count matches
    correct = 0
    format_correct = 0
    total = len(predictions)

    if total == 0:
        return {"accuracy": 0.0, "format_accuracy": 0.0}

    for pred, label in zip(predictions, labels):
        # Convert to strings if needed
        pred_str = str(pred).strip().upper() if not isinstance(pred, str) else pred.strip().upper()
        label_str = str(label).strip().upper() if not isinstance(label, str) else label.strip().upper()

        # Check if prediction is in correct format (A, B, C, or D)
        if re.match(r'^[A-D]$', pred_str):
            format_correct += 1

        # Check if answer is correct
        if pred_str == label_str:
            correct += 1

    return {
        "accuracy": correct / total,
        "format_accuracy": format_correct / total,
    }


def compute_saq_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute metrics for SAQ task.

    Args:
        eval_pred: Evaluation predictions from trainer.

    Returns:
        Dictionary with exact_match, bleu, rouge1, rougeL metrics.
    """
    predictions, labels = eval_pred

    if hasattr(predictions, 'predictions'):
        predictions = predictions.predictions

    total = len(predictions)
    if total == 0:
        return {
            "exact_match": 0.0,
            "bleu": 0.0,
            "rouge1": 0.0,
            "rougeL": 0.0,
        }

    # Convert to strings
    pred_strs = [str(p).strip() if not isinstance(p, str) else p.strip() for p in predictions]
    label_strs = [str(l).strip() if not isinstance(l, str) else l.strip() for l in labels]

    # Exact match
    exact_matches = sum(1 for p, l in zip(pred_strs, label_strs) if p.lower() == l.lower())
    exact_match_score = exact_matches / total

    # BLEU and ROUGE scores
    bleu_score = 0.0
    rouge1_score = 0.0
    rougeL_score = 0.0

    try:
        import evaluate

        # BLEU
        bleu_metric = evaluate.load("sacrebleu")
        # sacrebleu expects references as list of lists
        references = [[ref] for ref in label_strs]
        bleu_result = bleu_metric.compute(predictions=pred_strs, references=references)
        bleu_score = bleu_result["score"] / 100.0  # Normalize to 0-1

        # ROUGE
        rouge_metric = evaluate.load("rouge")
        rouge_result = rouge_metric.compute(predictions=pred_strs, references=label_strs)
        rouge1_score = rouge_result["rouge1"]
        rougeL_score = rouge_result["rougeL"]

    except Exception:
        # If evaluate fails, return zeros for these metrics
        pass

    return {
        "exact_match": exact_match_score,
        "bleu": bleu_score,
        "rouge1": rouge1_score,
        "rougeL": rougeL_score,
    }


def log_lora_params(lora_config: Any) -> Dict[str, Any]:
    """Extract LoRA parameters for logging.

    Args:
        lora_config: LoRA configuration object.

    Returns:
        Dictionary of LoRA parameters.
    """
    params = {}

    if hasattr(lora_config, 'r'):
        params['lora_r'] = lora_config.r
    if hasattr(lora_config, 'lora_alpha'):
        params['lora_alpha'] = lora_config.lora_alpha
    if hasattr(lora_config, 'lora_dropout'):
        params['lora_dropout'] = lora_config.lora_dropout
    if hasattr(lora_config, 'target_modules'):
        modules = lora_config.target_modules
        if isinstance(modules, (list, tuple)):
            params['lora_target_modules'] = ','.join(modules)
        else:
            params['lora_target_modules'] = str(modules)

    return params


def get_trainable_params(model: torch.nn.Module) -> Dict[str, Any]:
    """Get trainable parameters info from model.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with trainable_params and trainable_params_percent.
    """
    trainable_params = 0
    all_params = 0

    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_percent = 100 * trainable_params / all_params if all_params > 0 else 0

    return {
        "trainable_params": trainable_params,
        "total_params": all_params,
        "trainable_params_percent": trainable_percent,
    }


class MLflowSystemMetricsCallback(TrainerCallback):
    """Callback for logging system metrics to MLflow."""

    def __init__(self, log_every_n_steps: int = 10):
        """Initialize the callback.

        Args:
            log_every_n_steps: How often to log system metrics.
        """
        self.log_every_n_steps = log_every_n_steps
        self._step_start_time = None
        self._tokens_processed = 0

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Record step start time."""
        import time
        self._step_start_time = time.time()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Log system metrics at the end of each step."""
        if state.global_step % self.log_every_n_steps != 0:
            return

        try:
            import mlflow
            import time

            metrics = {}

            # GPU metrics
            if torch.cuda.is_available():
                metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
                metrics["gpu_memory_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

            # Tokens per second (approximate)
            if self._step_start_time is not None:
                step_time = time.time() - self._step_start_time
                batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
                # Approximate tokens: batch_size * max_seq_length (use 512 as default)
                max_seq_length = getattr(args, 'max_seq_length', 512)
                tokens_in_step = batch_size * max_seq_length
                if step_time > 0:
                    metrics["tokens_per_second"] = tokens_in_step / step_time

            if metrics:
                mlflow.log_metrics(metrics, step=state.global_step)

        except Exception:
            # Silently ignore MLflow errors
            pass

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """Log initial model parameters."""
        try:
            import mlflow

            if model is not None:
                params_info = get_trainable_params(model)
                mlflow.log_params({
                    "trainable_params": params_info["trainable_params"],
                    "total_params": params_info["total_params"],
                    "trainable_params_percent": f"{params_info['trainable_params_percent']:.2f}",
                })
        except Exception:
            pass


def setup_mlflow(
    tracking_uri: str,
    experiment_name: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Optional[Any]:
    """Setup MLflow tracking.

    Args:
        tracking_uri: MLflow tracking URI.
        experiment_name: Name of the experiment.
        run_name: Optional name for this run.
        tags: Optional tags for the run.

    Returns:
        MLflow run object or None if setup fails.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        run = mlflow.start_run(run_name=run_name, tags=tags)
        return run

    except Exception as e:
        print(f"Warning: Failed to setup MLflow: {e}")
        return None


def end_mlflow_run():
    """End the current MLflow run."""
    try:
        import mlflow
        mlflow.end_run()
    except Exception:
        pass


def log_params_to_mlflow(params: Dict[str, Any]):
    """Log parameters to MLflow.

    Args:
        params: Dictionary of parameters to log.
    """
    try:
        import mlflow

        # Convert all values to strings (MLflow requirement)
        str_params = {}
        for k, v in params.items():
            if isinstance(v, (list, tuple)):
                str_params[k] = ','.join(str(x) for x in v)
            else:
                str_params[k] = str(v)

        mlflow.log_params(str_params)
    except Exception as e:
        print(f"Warning: Failed to log params to MLflow: {e}")


def log_artifact_to_mlflow(local_path: str, artifact_path: Optional[str] = None):
    """Log an artifact to MLflow.

    Args:
        local_path: Local path to the artifact.
        artifact_path: Optional path within the artifact store.
    """
    try:
        import mlflow
        mlflow.log_artifact(local_path, artifact_path)
    except Exception as e:
        print(f"Warning: Failed to log artifact to MLflow: {e}")
