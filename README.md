# QA Model - LoRA Fine-Tuning

LoRA fine-tuning system for Multiple Choice Questions (MCQ) and Short Answer Questions (SAQ) tasks with Hydra configuration management.

## Project Structure

```
qa-model/
├── configs/
│   ├── config.yaml                 # Main Hydra config
│   ├── model/
│   │   ├── mistral_7b.yaml        # Mistral-7B config
│   │   └── llama3_8b.yaml         # Llama-3-8B config
│   ├── lora/
│   │   └── default.yaml           # LoRA hyperparameters
│   ├── training/
│   │   ├── mcq.yaml               # MCQ training config
│   │   └── saq.yaml               # SAQ training config
│   └── inference/
│       └── default.yaml           # Inference settings
├── src/qa_model/
│   ├── prompts.py                 # Prompt templates
│   ├── data/
│   │   ├── preprocessing.py       # Data loading & splitting
│   │   └── dataset.py             # MCQDataset, SAQDataset classes
│   ├── models/
│   │   ├── loader.py              # Model loading with LoRA
│   │   └── lora_config.py         # LoRA configuration
│   ├── training/
│   │   └── trainer.py             # SFT training logic
│   └── inference/
│       ├── generator.py           # Text generation with stop tokens
│       ├── validator.py           # Format validation (MCQ/SAQ)
│       └── router.py              # Adapter routing & retry logic
├── scripts/
│   ├── train.py                   # Training entry point
│   └── infer.py                   # Inference entry point
├── data/
│   ├── train_dataset_mcq.csv      # MCQ training data
│   ├── train_dataset_saq.csv      # SAQ training data
│   ├── test_dataset_mcq.csv       # MCQ test data
│   └── test_dataset_saq.csv       # SAQ test data
└── notebooks/
    └── base_3models.ipynb         # Reference notebook
```

## Installation

```bash
pip install -e .
```

## Usage Examples

### Training

```bash
# Train MCQ adapter (with Mistral-7B by default)
python scripts/train.py task=mcq training=mcq

# Train SAQ adapter
python scripts/train.py task=saq training=saq

# Use Llama-3-8B instead
python scripts/train.py task=mcq model=llama3_8b

# Override hyperparameters
python scripts/train.py task=mcq training.epochs=4 lora.r=8

# Override multiple parameters
python scripts/train.py task=mcq training.epochs=4 lora.r=8 training.learning_rate=5e-5
```

### Inference

```bash
# Run MCQ inference
python scripts/infer.py task=mcq mode=infer

# Run SAQ inference
python scripts/infer.py task=saq mode=infer

# Disable validation/retry
python scripts/infer.py task=mcq inference.validation.enabled=false

# Enable stop tokens
python scripts/infer.py task=saq inference.use_stop_tokens=true
```

## Configuration Guide

This guide explains each configuration parameter, what it controls, and how to tune it. Designed for ML beginners.

---

### Understanding LoRA (Low-Rank Adaptation)

**What is LoRA?**

Instead of updating all billions of parameters in a large model (expensive!), LoRA adds small trainable "adapter" layers. Think of it like adding a small plugin to a large application - the original app stays unchanged, but the plugin modifies its behavior.

**File:** `configs/lora/default.yaml`

#### `r` (rank) — Default: `16`

**What it is:** The "size" of the adapter. Technically, the dimensionality of the low-rank matrices.

**Analogy:** Imagine summarizing a book. `r=4` is like a tweet-sized summary, `r=64` is like a detailed chapter-by-chapter summary. Higher rank = more capacity to learn patterns.

**How it affects training:**
- **Lower r (4-8):** Faster training, less memory, but may underfit (not learn enough)
- **Higher r (32-64):** Slower training, more memory, but can learn more complex patterns

**When to change:**
- Increase if model isn't learning well (validation loss stays high)
- Decrease if you're running out of GPU memory
- Start with 16, adjust based on results

```bash
python scripts/train.py lora.r=8      # Memory-constrained
python scripts/train.py lora.r=32     # Complex task needs more capacity
```

---

#### `alpha` — Default: `32`

**What it is:** A scaling factor that controls how much the LoRA adapter influences the model's output.

**The math:** The adapter's contribution is multiplied by `alpha/r`. With defaults (alpha=32, r=16), the scaling is 2x.

**Analogy:** Think of it as a volume knob for the adapter. Higher alpha = louder adapter voice relative to the base model.

**Rule of thumb:** Keep `alpha = 2 × r` as a starting point. This gives a scaling factor of 2.

**When to change:**
- If adapter changes are too subtle, increase alpha
- If model outputs become erratic, decrease alpha
- Usually change alongside `r` to maintain the ratio

```bash
python scripts/train.py lora.r=8 lora.alpha=16    # Maintains 2x scaling
python scripts/train.py lora.r=32 lora.alpha=64   # Maintains 2x scaling
```

---

#### `dropout` — Default: `0.05`

**What it is:** During training, randomly "turns off" 5% of the adapter connections each step.

**Why it exists:** Prevents overfitting. Forces the model to not rely too heavily on any single connection, making it more robust.

**Analogy:** Like training a team where random members are absent each day - the team learns to function without depending on any single person.

**When to change:**
- **Increase (0.1-0.2):** If model overfits (training loss low, validation loss high)
- **Decrease (0.01-0.03):** If model underfits or you have lots of training data
- **Set to 0:** For very large datasets where overfitting isn't a concern

```bash
python scripts/train.py lora.dropout=0.1    # More regularization
python scripts/train.py lora.dropout=0.0    # No regularization
```

---

#### `target_modules` — Default: `[q_proj, k_proj, v_proj, o_proj]`

**What it is:** Which parts of the transformer to attach LoRA adapters to.

**What these modules are:**
- `q_proj`, `k_proj`, `v_proj`: The Query, Key, Value projections in attention (how the model "pays attention" to different parts of the input)
- `o_proj`: Output projection of attention

**Analogy:** Choosing which instruments to modify in an orchestra. Attention layers are like the conductors - modifying them affects how the whole orchestra plays together.

**When to change:**
- The defaults target attention layers, which is usually sufficient
- For more aggressive fine-tuning, you can add MLP layers: `gate_proj`, `up_proj`, `down_proj`
- More modules = more trainable parameters = more memory needed

---

### Understanding Training Parameters

**File:** `configs/training/mcq.yaml` and `configs/training/saq.yaml`

#### `epochs` — Default: `3`

**What it is:** How many times the model sees the entire training dataset.

**Analogy:** Reading a textbook. Epoch 1 = first read-through, Epoch 2 = second read-through, etc. Each pass helps reinforce learning.

**How it affects training:**
- **Too few epochs:** Model hasn't learned enough (underfitting)
- **Too many epochs:** Model memorizes training data instead of generalizing (overfitting)

**Signs you need more epochs:**
- Validation loss is still decreasing when training ends
- Model performs poorly on test data

**Signs you need fewer epochs:**
- Validation loss starts increasing while training loss keeps decreasing
- Model performs great on training data but poorly on new data

```bash
python scripts/train.py training.epochs=5    # Train longer
python scripts/train.py training.epochs=1    # Quick test run
```

---

#### `batch_size` — Default: `4`

**What it is:** How many examples the model processes before updating its weights.

**Analogy:** A teacher grading papers. Batch size 1 = grade one paper, give feedback, repeat. Batch size 32 = grade 32 papers, then give overall feedback.

**Trade-offs:**
- **Larger batches (8-32):** More stable gradients, faster training (if memory allows), may generalize worse
- **Smaller batches (1-4):** Noisier gradients (can help escape bad solutions), slower, uses less memory

**Memory impact:** Batch size is the #1 factor in GPU memory usage. If you get "CUDA out of memory", reduce batch size first.

```bash
python scripts/train.py training.batch_size=2    # Less memory
python scripts/train.py training.batch_size=8    # More memory, faster
```

---

#### `gradient_accumulation_steps` — Default: `4`

**What it is:** A trick to simulate larger batches without using more memory.

**How it works:** Instead of updating weights after each batch, accumulate gradients over N batches, then update.

**Effective batch size = batch_size × gradient_accumulation_steps**

With defaults: 4 × 4 = 16 effective batch size

**When to use:**
- You want larger effective batch sizes but can't fit them in memory
- Example: Can't fit batch_size=16, so use batch_size=4 with gradient_accumulation_steps=4

```bash
# Both give effective batch size of 32:
python scripts/train.py training.batch_size=8 training.gradient_accumulation_steps=4
python scripts/train.py training.batch_size=4 training.gradient_accumulation_steps=8  # Less memory
```

---

#### `learning_rate` — Default: `1e-4` (0.0001)

**What it is:** How big of a step to take when updating model weights.

**Analogy:** Walking toward a destination. High learning rate = big steps (faster but might overshoot). Low learning rate = small steps (slower but more precise).

**This is the most important hyperparameter to tune.**

**How it affects training:**
- **Too high (1e-3+):** Training is unstable, loss spikes or goes to NaN
- **Too low (1e-6):** Training is very slow, model barely learns
- **Just right:** Loss decreases smoothly

**Typical ranges for LoRA fine-tuning:**
- `1e-4` to `5e-4`: Aggressive learning (default, good for small datasets)
- `1e-5` to `5e-5`: Conservative learning (good for larger datasets)
- `1e-6`: Very conservative (rarely needed)

```bash
python scripts/train.py training.learning_rate=5e-5    # More conservative
python scripts/train.py training.learning_rate=2e-4    # More aggressive
```

---

#### `warmup_ratio` — Default: `0.1`

**What it is:** Fraction of training spent gradually increasing the learning rate from 0 to the target value.

**Why it matters:** Starting with a high learning rate can destabilize training. Warmup lets the model "settle in" before aggressive learning.

**With defaults:** 10% of steps are warmup. For 1000 total steps, first 100 steps gradually increase LR.

**When to change:**
- **Increase (0.2):** If training is unstable at the start
- **Decrease (0.05):** If you have limited data and need all steps for actual learning
- **Set to 0:** Skip warmup entirely (risky but sometimes works)

```bash
python scripts/train.py training.warmup_ratio=0.2    # Longer warmup
python scripts/train.py training.warmup_ratio=0.0    # No warmup
```

---

#### `scheduler` — Default: `cosine`

**What it is:** How the learning rate changes during training after warmup.

**Options:**
- **`cosine`** (default): LR smoothly decreases following a cosine curve. Gentle decay, popular choice.
- **`linear`**: LR decreases in a straight line to 0. Simpler but effective.
- **`constant`**: LR stays the same throughout. Use with caution.

**Visual:**
```
cosine:   ‾‾‾\___    (smooth decay)
linear:   ‾‾‾\       (straight decay)
constant: ‾‾‾‾‾‾‾    (no decay)
```

**When to change:**
- `cosine` is usually best for fine-tuning
- `linear` if cosine doesn't work well
- `constant` almost never used for fine-tuning

```bash
python scripts/train.py training.scheduler=linear
```

---

#### `val_ratio` — Default: `0.15`

**What it is:** Fraction of training data held out for validation (not used for training).

**Why it matters:** Validation data lets you monitor if the model is overfitting. The model never sees this data during training, so validation loss tells you how well it generalizes.

**Trade-offs:**
- **Higher (0.2-0.3):** More reliable validation metrics, but less training data
- **Lower (0.05-0.1):** More training data, but validation metrics are noisier

**When to change:**
- Small dataset (< 1000 examples): Use lower ratio (0.1) to keep more for training
- Large dataset (> 10000 examples): Can use higher ratio (0.2) for reliable validation

```bash
python scripts/train.py training.val_ratio=0.1    # More training data
python scripts/train.py training.val_ratio=0.2    # More validation data
```

---

#### `max_seq_length` — Default: `512`

**What it is:** Maximum number of tokens (roughly words/subwords) in input sequences.

**What happens to longer sequences:** They get truncated (cut off).

**Memory impact:** Longer sequences use significantly more memory (scales quadratically with attention).

**When to change:**
- Increase if your questions/answers are being cut off
- Decrease if running out of memory

```bash
python scripts/train.py training.max_seq_length=256     # Shorter, less memory
python scripts/train.py training.max_seq_length=1024    # Longer, more memory
```

---

#### `fp16` vs `bf16` — Defaults: `fp16=true`, `bf16=false`

**What they are:** Precision formats for storing numbers during training.

- **fp16 (float16):** Half precision, works on most GPUs
- **bf16 (bfloat16):** Alternative half precision, only on newer GPUs (Ampere+, like A100, RTX 30/40 series)

**Why use them:** Training in half precision uses ~half the memory and is faster.

**Which to choose:**
- Most setups: Keep `fp16=true`
- If you have Ampere+ GPU and see NaN losses: Try `bf16=true, fp16=false`

```bash
python scripts/train.py training.fp16=false training.bf16=true    # For newer GPUs
```

---

#### Logging and Checkpointing

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `logging_steps` | `10` | Print loss every N steps |
| `save_steps` | `100` | Save checkpoint every N steps |
| `eval_steps` | `100` | Run validation every N steps |
| `save_total_limit` | `2` | Keep only N most recent checkpoints (saves disk space) |

**When to change:**
- Increase `logging_steps` if output is too verbose
- Decrease `eval_steps` if you want to catch overfitting early
- Increase `save_total_limit` if you want to keep more checkpoints for comparison

---

### Understanding Inference Parameters

**File:** `configs/inference/default.yaml`

#### `max_new_tokens` — Default: `64`

**What it is:** Maximum number of tokens the model can generate in response.

**For MCQ:** Usually need only 1-5 tokens (the answer letter)
**For SAQ:** May need 50-200 tokens for full explanations

```bash
python scripts/infer.py inference.max_new_tokens=10     # Short answers
python scripts/infer.py inference.max_new_tokens=256    # Long answers
```

---

#### `do_sample` — Default: `false`

**What it is:** Whether to introduce randomness when generating text.

- **`false` (greedy):** Always pick the most likely next token. Deterministic - same input always gives same output.
- **`true` (sampling):** Randomly select tokens based on their probabilities. Non-deterministic - outputs vary.

**When to use sampling:**
- You want diverse/creative outputs
- You're generating multiple answers to compare

**When to use greedy:**
- You want consistent, reproducible results
- For MCQ where there's one correct answer

```bash
python scripts/infer.py inference.do_sample=true    # Enable randomness
```

---

#### `temperature` — Default: `0.0`

**What it is:** Controls randomness when `do_sample=true`. Only matters if sampling is enabled.

**How it works:**
- **Temperature = 0:** Same as greedy (no randomness)
- **Temperature = 0.1-0.5:** Slightly random, still mostly predictable
- **Temperature = 0.7-1.0:** Moderately random, good for creative tasks
- **Temperature > 1.0:** Very random, often incoherent

**Analogy:** Temperature of water. Cold (0) = frozen/rigid. Warm (0.7) = flowing. Hot (1.5) = chaotic boiling.

```bash
python scripts/infer.py inference.do_sample=true inference.temperature=0.7
```

---

#### `top_p` (nucleus sampling) — Default: `1.0`

**What it is:** An alternative way to control randomness. Only considers tokens whose cumulative probability is within `top_p`.

**How it works:**
- `top_p=1.0`: Consider all tokens (no filtering)
- `top_p=0.9`: Consider tokens that make up 90% of probability mass

**Analogy:** Instead of inviting everyone to a party (1.0), only invite the top 90% most likely guests (0.9).

**When to use:**
- Often used with `temperature=1.0` for controlled diversity
- `top_p=0.9` is a common choice for balanced generation

```bash
python scripts/infer.py inference.do_sample=true inference.top_p=0.9 inference.temperature=1.0
```

---

#### Validation Settings

**`validation.enabled`** — Default: `true`

Checks if model output matches expected format (e.g., MCQ should output A/B/C/D).

**`validation.max_retries`** — Default: `2`

If output is invalid, regenerate up to N times.

**When to disable:**
- During development/debugging to see raw outputs
- If validation is too strict and rejecting valid answers

```bash
python scripts/infer.py inference.validation.enabled=false    # See raw outputs
python scripts/infer.py inference.validation.max_retries=5    # Try harder for valid output
```

---

### Quick Reference: Common Scenarios

**"Model isn't learning well"**
```bash
# Try: more epochs, higher LoRA rank, lower learning rate
python scripts/train.py training.epochs=5 lora.r=32 training.learning_rate=5e-5
```

**"Running out of GPU memory"**
```bash
# Try: smaller batch, lower LoRA rank, shorter sequences
python scripts/train.py training.batch_size=2 lora.r=8 training.max_seq_length=256
```

**"Model is overfitting"**
```bash
# Try: fewer epochs, more dropout, more validation data
python scripts/train.py training.epochs=2 lora.dropout=0.1 training.val_ratio=0.2
```

**"Training is unstable (loss spikes)"**
```bash
# Try: lower learning rate, longer warmup
python scripts/train.py training.learning_rate=5e-5 training.warmup_ratio=0.2
```

**"Want diverse outputs during inference"**
```bash
python scripts/infer.py inference.do_sample=true inference.temperature=0.7
```

## Adapters

Trained adapters are saved to `outputs/adapters/`:
- `adapter_mcq/` - MCQ adapter weights
- `adapter_saq/` - SAQ adapter weights

## Models

Pre-cached models are available at `/data/cat/ws/albu670g-qa-model/models`:
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Meta-Llama-3-8B-Instruct`
