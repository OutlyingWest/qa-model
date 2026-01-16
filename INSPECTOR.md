# Task Description (Agent Request)

Your task is to validate the factual correctness of my models’ answers.  
Only MCQ questions must be checked.

## 1. Validation of MCQ Answers

**Path to the file with questions:**  
`data/test_dataset_mcq.csv`

**Path to the answers of Model #1:**  
`results/llama_3_8b_instruct/mcq_submission__meta_llama__meta_llama_3_8b_instruct.tsv`

**Path to the answers of Model #2:**  
`results/mistral_7b_instruct/mcq_submission__mistralai__mistral_7b_instruct_v0.2.tsv`

## Output Requirements

The validation results for each question must be provided in `.tsv` format, consistent with the other files.

The output file must contain **2 columns**:

- `MCQID`
- `conclusion`

For each question, the `conclusion` column must contain **strictly one** of the following values:

- `True` — if you consider the model’s answer to be correct
- `False` — if you consider the model’s answer to be incorrect and believe another answer is correct

Always output **only one** of the two options: `True` or `False`.

Use Web Search for validation.

You must put the validation results to the corresponding to each model directories:
- For model №1: `results/llama_3_8b_instruct/claude_validated_llama_3_8b_instruct.tsv`
- For model №2: `results/mistral_7b_instruct/claude_validated_mistral_7b_instruct.tsv`