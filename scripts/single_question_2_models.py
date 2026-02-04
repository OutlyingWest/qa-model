#!/usr/bin/env python3
"""Single-question inference using two base models."""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_IDS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    """Build a model-appropriate prompt."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"System: {messages[0]['content']}\nUser: {question}\nAssistant:"


def ask_model(
    model,
    tokenizer: AutoTokenizer,
    device: str,
    question: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Run a single-question generation and return the model answer."""
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    pad_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask one question to a base model (Mistral or Llama-3)."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_IDS.keys()),
        default="mistral",
        help="Which base model to use.",
    )
    parser.add_argument(
        "--question",
        default="What is the capital of France?",
        help="Question to ask the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy).",
    )
    parser.add_argument(
        "--cache-dir",
        default="/data/cat/ws/albu670g-qa-model/models",
        help="Hugging Face cache directory.",
    )
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    if not os.path.isdir(cache_dir):
        cache_dir = os.path.abspath("../models")

    model_id = MODEL_IDS[args.model]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    answer = ask_model(
        model,
        tokenizer,
        device,
        args.question,
        args.max_new_tokens,
        args.temperature,
    )
    print(answer)


if __name__ == "__main__":
    main()
