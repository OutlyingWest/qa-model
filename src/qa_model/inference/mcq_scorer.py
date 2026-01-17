"""MCQ scoring utilities (logprob scoring + country-aware reranking)."""

from __future__ import annotations

import ast
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def _normalize_option_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _parse_mapping_str(value: str) -> Dict[str, str]:
    """Parse a dict-like string from the datasets (JSON-ish or Python literal)."""
    value = value.strip()
    if not value:
        return {}
    # Most files use a JSON-like dict with quotes/newlines; ast handles it reliably here.
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        pass
    # Best-effort fallback: try JSON.
    try:
        import json

        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        pass
    return {}


@dataclass(frozen=True)
class MCQLogprobConfig:
    choice_letters: Tuple[str, str, str, str] = ("A", "B", "C", "D")
    variants: Tuple[str, str, str] = (" {choice}", "\n{choice}", "{choice}")


class CountryPriorReranker:
    """Adds a learned prior bonus based on option_text -> likely country mapping from MCQ train."""

    def __init__(
        self,
        text_logp_by_country: Dict[str, Dict[str, float]],
        target_countries: Sequence[str],
    ) -> None:
        self._text_logp_by_country = text_logp_by_country
        self._target_countries = tuple(target_countries)
        self._uniform_logp = -math.log(len(self._target_countries)) if self._target_countries else 0.0

    @classmethod
    def from_train_csv(
        cls,
        train_csv_path: Union[str, Path],
        target_countries: Sequence[str] = ("US", "UK", "China", "Iran"),
        alpha: float = 1.0,
    ) -> "CountryPriorReranker":
        """Build reranker from `data/train_dataset_mcq.csv` (uses `choices` + `choice_countries`)."""
        train_csv_path = Path(train_csv_path)
        counts: Dict[str, Dict[str, int]] = {}

        with train_csv_path.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                choices = _parse_mapping_str(row.get("choices", ""))
                choice_countries = _parse_mapping_str(row.get("choice_countries", ""))
                for letter, option_text in choices.items():
                    tag = choice_countries.get(letter)
                    if not tag:
                        continue
                    norm_text = _normalize_option_text(option_text)
                    if norm_text not in counts:
                        counts[norm_text] = {}
                    counts[norm_text][tag] = counts[norm_text].get(tag, 0) + 1

        text_logp_by_country: Dict[str, Dict[str, float]] = {}
        target_countries = tuple(target_countries)
        for text, tag_counts in counts.items():
            total = sum(tag_counts.values())
            denom = total + alpha * len(target_countries)
            if denom <= 0:
                continue
            per_country: Dict[str, float] = {}
            for country in target_countries:
                p = (tag_counts.get(country, 0) + alpha) / denom
                per_country[country] = math.log(p)
            text_logp_by_country[text] = per_country

        return cls(text_logp_by_country=text_logp_by_country, target_countries=target_countries)

    def bonus(self, option_text: str, target_country: str) -> float:
        """Returns a centered log-prior bonus for this option text under the target country."""
        norm_text = _normalize_option_text(option_text)
        logp = self._text_logp_by_country.get(norm_text, {}).get(target_country)
        if logp is None:
            return 0.0
        # Center relative to uniform so bonuses are comparable across countries.
        return logp - self._uniform_logp


@torch.inference_mode()
def _logprob_of_completion_from_cache(
    model: PreTrainedModel,
    past_key_values,
    next_logprobs: torch.FloatTensor,
    completion_token_ids: List[int],
) -> float:
    """Compute log P(completion | prompt) starting from an existing KV-cache."""
    if not completion_token_ids:
        return float("-inf")

    total_logp = 0.0
    past = past_key_values
    lp_next = next_logprobs
    for token_id in completion_token_ids:
        total_logp += float(lp_next[0, token_id])

        step_ids = torch.tensor([[token_id]], device=model.device, dtype=torch.long)
        step_out = model(input_ids=step_ids, past_key_values=past, use_cache=True)
        past = step_out.past_key_values
        lp_next = torch.log_softmax(step_out.logits[:, -1, :], dim=-1)

    return total_logp


@torch.inference_mode()
def choose_mcq_via_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    *,
    logprob_cfg: Optional[MCQLogprobConfig] = None,
    mcq_choices: Optional[Dict[str, str]] = None,
    target_country: Optional[str] = None,
    reranker: Optional[CountryPriorReranker] = None,
    rerank_weight: float = 0.0,
) -> str:
    """Pick A/B/C/D by scoring token-level logprobs, optionally adding a country prior bonus."""
    cfg = logprob_cfg or MCQLogprobConfig()

    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_input_ids = encoded["input_ids"]
    prompt_attention_mask = encoded.get("attention_mask", torch.ones_like(prompt_input_ids))

    prompt_out = model(
        input_ids=prompt_input_ids.to(model.device),
        attention_mask=prompt_attention_mask.to(model.device),
        use_cache=True,
    )
    base_past_key_values = prompt_out.past_key_values
    base_next_logprobs = torch.log_softmax(prompt_out.logits[:, -1, :], dim=-1)

    scores: Dict[str, float] = {}
    for choice in cfg.choice_letters:
        best_lp = float("-inf")
        for variant_tmpl in cfg.variants:
            variant = variant_tmpl.format(choice=choice)
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            lp = _logprob_of_completion_from_cache(
                model=model,
                past_key_values=base_past_key_values,
                next_logprobs=base_next_logprobs,
                completion_token_ids=token_ids,
            )
            if lp > best_lp:
                best_lp = lp

        score = best_lp
        if (
            reranker is not None
            and rerank_weight
            and target_country
            and mcq_choices is not None
            and choice in mcq_choices
        ):
            score += float(rerank_weight) * reranker.bonus(mcq_choices[choice], target_country)

        scores[choice] = score

    # Deterministic tie-breaker: A > B > C > D by order in cfg.choice_letters.
    best_choice = max(cfg.choice_letters, key=lambda c: (scores.get(c, float("-inf")), -cfg.choice_letters.index(c)))
    return best_choice
