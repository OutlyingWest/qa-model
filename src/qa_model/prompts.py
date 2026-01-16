"""Prompt templates for MCQ and SAQ tasks."""

SAQ_SYSTEM_PROMPT = """Provide ONE word answer to the given question.

Give the answer in the following format:
Answer: *provided answer*.
Explanation: *provided explanation*.

If no answer can be provided:
Answer: idk.
Explanation: *provided explanation*."""

MCQ_SYSTEM_PROMPT = """Answer the multiple choice question.
Pick only one option (A, B, C, or D) without explanation."""


def build_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Build a model-appropriate prompt using the tokenizer's chat template."""
    chat_template = getattr(tokenizer, "chat_template", None)

    if chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback when no chat template exists (e.g., base models)
    return f"[INST]{system_prompt.strip()}\n{user_prompt.strip()}[/INST]"


def build_mcq_prompt(tokenizer, question: str) -> str:
    """Build a prompt for MCQ task."""
    return build_prompt(tokenizer, MCQ_SYSTEM_PROMPT, question)


def build_saq_prompt(tokenizer, question: str) -> str:
    """Build a prompt for SAQ task."""
    return build_prompt(tokenizer, SAQ_SYSTEM_PROMPT, f"Question: {question}")


def format_saq_response(answer: str) -> str:
    """Format a SAQ response for training."""
    return f"Answer: {answer}.\nExplanation: Based on the question context."


def format_mcq_response(choice: str) -> str:
    """Format a MCQ response for training."""
    return choice.upper()
