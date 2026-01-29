## Default model
### MCQ Results

|Metric|Score|
|---|---|
|Overall|   |
|Accuracy|0.74|
|By Country|   |
|China|0.66|
|Iran|0.62|
|UK|0.90|
|US|0.80|

### SAQ Results

| Metric     | Score |
| ---------- | ----- |
| Overall    |       |
| Accuracy   | 0.50  |
| By Country |       |
| CN         | 0.40  |
| GB         | 0.59  |
| IR         | 0.38  |
| US         | 0.64  |


## Experiments, results and improvements - for SAQ and MCQ in the corresponding sections

### SAQ
#### Parsing and prompt refined

After updating the prompt, adding multiword support to `format_saq_response()` / `parse_saq_response()`, and retraining the LoRA on the new format, an accuracy gain of approximately 8.0% was achieved.

#### refined validation retries
Only 4 saq answers affected - insignificant improvement possible

#### saq: gate_proj retrained

LoRA for SAQ retrained with gate_proj layer included in training - ~1% accuracy improvement

#### RAG for SAQ
1. Raw
2. stop-words removed
3. Stemming

Использование RAG на сырых вопросах только ухудшило процент правильных ответов. Количество ответов idk выросло и было заметно даже невооруженным глазом. После первоначального эксперимента из индекса были удалены стоп-слова и произведен стемминг.

#### RAG for SAQ future experiment
only add a stub here

### MCQ

#### logprob: w1.4
Gave ~ 4%

##### 2) Чёткий алгоритм

##### A) Logprob-scoring (выбор A/B/C/D по вероятности продолжения)

Вход: prompt (ваш chat-пrompt, заканчивается Answer:), модель+токенайзер.

1. Токенизировать prompt → input_ids, attention_mask.
2. Один forward pass модели на prompt с use_cache=True:
    - получаем past_key_values (KV-cache)
    - получаем logits последнего шага → это распределение для следующего токена.
3. Для каждого кандидата choice ∈ {A,B,C,D}:
    - перебрать несколько текстовых вариантов продолжения (по умолчанию): " A", "\nA", "A" (аналогично для B/C/D)
    - закодировать вариант в completion_token_ids
    - посчитать log P(completion | prompt):
        - взять logprob первого токена из распределения “следующий токен”
        - затем по одному токену “докармливать” модель, используя past_key_values, каждый раз суммируя logprob текущего токена
    - взять максимум logprob по вариантам (" A" vs "\nA" vs "A") → это score для буквы.
4. Выбрать букву с максимальным score.

##### B) Country-aware rerank (бонус за “свою” опцию)

Два этапа: подготовка prior и применение.

Подготовка prior (один раз на запуск):

1. Прочитать data/train_dataset_mcq.csv.
2. Для каждого train-примера:
    - распарсить choices (тексты A/B/C/D)
    - распарсить choice_countries (страна-тег каждой опции)
    - для каждого option_text накопить счётчик count(option_text, tag_country) += 1
3. Превратить счётчики в log P(tag_country | option_text) со сглаживанием (add-α).

Применение на test-вопросе:

4. Для каждой опции взять её текст option_text из choices.
5. Взять target_country из столбца country.
6. Вычислить bonus = logP(target_country|option_text) - logP_uniform (центрирование).
7. Итоговый score буквы:
    - final_score = logprob_score + rerank_weight * bonus
8. Выбрать максимум.

## List of submissions
Your goal is to map this list of improvements to:
1. Section "Experiments, results and improvements - for SAQ and MCQ in the corresponding sections" of this document
2. Git commits that contain corresponding improvements (don't provide code snippets but understand changes in commits and reflect it in the report)

492313	mcq_logprob_w2.0.zip	2026-01-17 21:02	Finished	0.79		
492309	mcq_logprob_1.zip	2026-01-17 20:59	Finished	0.77		 
492303	mcq_logprob_w1.4.zip	2026-01-17 20:53	Finished	0.78		 
492244	saq_gate_proj_retrained.zip	2026-01-17 20:02	Finished	0.74		 
492226	refined_validation_retries.zip	2026-01-17 19:42	Finished	0.74		 
492181	parsing_prompt_refined.zip	2026-01-17 18:54	Finished	0.74		 
492152	refined_saq_prompt_16.zip	2026-01-17 18:16	Finished	0.74		 
492151	refined_saq_prompt_24.zip	2026-01-17 18:14	Finished	0.74		 
491415	lora_submission.zip	2026-01-17 01:08	Finished	0.74


## The best result achieved
The best result is achieved combining all mentioned improvements (without RAG) you can mention
the last RAG experiment and its result along with the reasons behind its failure and present this direction as a direction for future improvements 

### MCQ Results

|Metric|Score|
|---|---|
|Overall|   |
|Accuracy|0.79|
|By Country|   |
|China|0.74|
|Iran|0.68|
|UK|0.91|
|US|0.84|

### SAQ Results

|Metric|Score|
|---|---|
|Overall|   |
|Accuracy|0.59|
|By Country|   |
|CN|0.57|
|GB|0.66|
|IR|0.43|
|US|0.71|