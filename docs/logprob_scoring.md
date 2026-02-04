# Logprob Scoring: Полное объяснение

## 1. Что выдаёт языковая модель?

Языковая модель (LLM) — это по сути **предсказатель следующего слова**. На каждом шаге она получает текст и выдаёт **распределение вероятностей** по всему словарю.

### Словарь (Vocabulary)
Модель работает не со словами, а с **токенами** — кусочками слов. Например, слово "невероятный" может быть разбито на токены: `["не", "вер", "оятный"]`.

Словарь — это фиксированный набор всех возможных токенов. Типичный размер: 32,000 – 128,000 токенов.

### Logits — сырой выход модели
Когда модель обрабатывает текст, на выходе последнего слоя она выдаёт **logits** — вектор чисел размером с весь словарь:

$$\text{logits} = [z_1, z_2, z_3, \ldots, z_V]$$

где $V$ — размер словаря.

**Logits** — это "сырые баллы", которые модель присвоила каждому токену. Они могут быть любыми числами: положительными, отрицательными, большими, маленькими.

**Пример:**
```
Токен "A" → logit = 2.5
Токен "B" → logit = 1.2
Токен "C" → logit = -0.3
Токен "D" → logit = 0.8
```

---

## 2. Softmax: превращаем logits в вероятности

Logits — это не вероятности (они не в диапазоне [0,1] и не суммируются в 1). Чтобы получить вероятности, применяется функция **softmax**:

$$P(token_i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

Эта формула делает две вещи:
1. Экспонента $e^{z_i}$ превращает любое число в положительное
2. Деление на сумму нормализует всё так, чтобы сумма = 1

**Пример (продолжение):**
```
e^2.5 = 12.18
e^1.2 = 3.32
e^-0.3 = 0.74
e^0.8 = 2.23
─────────────
Сумма = 18.47

P("A") = 12.18 / 18.47 = 0.66 (66%)
P("B") = 3.32 / 18.47 = 0.18 (18%)
P("C") = 0.74 / 18.47 = 0.04 (4%)
P("D") = 2.23 / 18.47 = 0.12 (12%)
```

---

## 3. Log Probability (Logprob)

**Log probability** — это логарифм вероятности:

$$\log P(token_i) = \log \left( \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}} \right)$$

Это можно упростить до:

$$\log P(token_i) = z_i - \log \sum_{j=1}^{V} e^{z_j}$$

Эта операция называется **log_softmax** (строка 162 в коде):

```python
base_next_logprobs = torch.log_softmax(prompt_out.logits[:, -1, :], dim=-1)
```

### Зачем логарифм?

1. **Численная стабильность**: вероятности часто очень маленькие (0.00001), их произведение быстро уходит в ноль. Логарифмы — числа порядка -10, -20 — с ними удобнее работать.

2. **Произведение → сумма**: для последовательности токенов вероятность — это произведение:
   $$P(\text{последовательность}) = P(t_1) \cdot P(t_2) \cdot P(t_3)$$

   С логарифмами это становится суммой:
   $$\log P(\text{последовательность}) = \log P(t_1) + \log P(t_2) + \log P(t_3)$$

---

## 4. Алгоритм Logprob Scoring для MCQ

Теперь применим это к задаче выбора ответа A/B/C/D.

### Идея
Вместо того чтобы просить модель *сгенерировать* ответ, мы **измеряем**, какой вариант модель считает наиболее вероятным продолжением.

### Шаг 1: Прогоняем prompt через модель

```python
prompt_out = model(
    input_ids=prompt_input_ids.to(model.device),
    attention_mask=prompt_attention_mask.to(model.device),
    use_cache=True,
)
```

Модель обрабатывает весь вопрос и возвращает:
- `past_key_values` — **KV-cache** (закэшированные вычисления, чтобы не пересчитывать prompt заново)
- `logits[:, -1, :]` — logits для **следующего токена** после prompt

### Шаг 2: Для каждого варианта считаем log P(вариант | prompt)

```python
for choice in cfg.choice_letters:  # ["A", "B", "C", "D"]
    for variant_tmpl in cfg.variants:  # [" {choice}", "\n{choice}", "{choice}"]
        variant = variant_tmpl.format(choice=choice)  # " A", "\nA", "A"
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        lp = _logprob_of_completion_from_cache(...)
```

**Зачем несколько вариантов?** Токен "A" и токен " A" (с пробелом) — это **разные токены**! Модель могла научиться отвечать с пробелом или с новой строки. Мы пробуем все варианты и берём лучший.

### Шаг 3: Подсчёт logprob для многотокенного completion

Функция `_logprob_of_completion_from_cache` (строки 113-134):

```python
def _logprob_of_completion_from_cache(..., completion_token_ids: List[int]) -> float:
    total_logp = 0.0
    for token_id in completion_token_ids:
        total_logp += float(lp_next[0, token_id])  # Берём logprob нужного токена

        # Делаем один шаг вперёд для следующего токена
        step_out = model(input_ids=[[token_id]], past_key_values=past, use_cache=True)
        lp_next = torch.log_softmax(step_out.logits[:, -1, :], dim=-1)

    return total_logp
```

**Визуально:**

```
Prompt: "Какая столица Франции? A) Париж B) Лондон C) Берлин D) Мадрид\nОтвет:"
                                                                            ↓
                                                          logprobs для следующего токена
                                                                            ↓
                              ┌─────────────────────────────────────────────────────────┐
                              │  P(" A") = 0.65  →  log P = -0.43                       │
                              │  P(" B") = 0.15  →  log P = -1.90                       │
                              │  P(" C") = 0.12  →  log P = -2.12                       │
                              │  P(" D") = 0.08  →  log P = -2.53                       │
                              └─────────────────────────────────────────────────────────┘

                              Выбираем: argmax → " A" (Париж)
```

Если completion состоит из нескольких токенов (например, `[" ", "A"]`), то:

$$\log P(\text{" A"}) = \log P(\text{" "} | \text{prompt}) + \log P(\text{"A"} | \text{prompt}, \text{" "})$$

### Шаг 4: Выбор лучшего ответа

```python
best_choice = max(cfg.choice_letters,
                  key=lambda c: (scores.get(c, float("-inf")),
                                 -cfg.choice_letters.index(c)))
```

Берём вариант с максимальным logprob. При равенстве — приоритет A > B > C > D.

---

## 5. Полная формула

Для вопроса $q$ и вариантов $\{A, B, C, D\}$:

$$\text{answer} = \arg\max_{c \in \{A,B,C,D\}} \left[ \max_{v \in \text{variants}} \log P(v(c) | q) \right]$$

где $v(c)$ — это варианты написания буквы c: `" A"`, `"\nA"`, `"A"`.

А сам $\log P(v(c) | q)$ для последовательности токенов $[t_1, t_2, \ldots, t_n]$:

$$\log P(v(c) | q) = \sum_{i=1}^{n} \log P(t_i | q, t_1, \ldots, t_{i-1})$$

---

## 6. Диаграмма процесса

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ВХОДНЫЕ ДАННЫЕ                                  │
│  Prompt: "Вопрос: ... A) ... B) ... C) ... D) ...\nОтвет:"                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ШАГ 1: ENCODE PROMPT                               │
│  tokenizer(prompt) → [tok₁, tok₂, ..., tokₙ]                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ШАГ 2: FORWARD PASS                                  │
│  model(tokens) → logits + KV-cache                                          │
│  log_softmax(logits[:, -1, :]) → base_next_logprobs                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ШАГ 3: SCORE КАЖДОГО ВАРИАНТА                            │
│                                                                              │
│   Для "A":  пробуем " A", "\nA", "A"                                         │
│             score_A = max(logP(" A"), logP("\nA"), logP("A"))                │
│                                                                              │
│   Для "B":  пробуем " B", "\nB", "B"                                         │
│             score_B = max(logP(" B"), logP("\nB"), logP("B"))                │
│                                                                              │
│   ... и так для C, D                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ШАГ 4: ARGMAX                                        │
│                                                                              │
│   scores = {"A": -0.43, "B": -1.90, "C": -2.12, "D": -2.53}                  │
│   answer = argmax(scores) = "A"                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ВЫХОД: "A"                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Преимущества Logprob Scoring

| Аспект | Генерация | Logprob Scoring |
|--------|-----------|-----------------|
| Детерминизм | Может варьироваться (sampling) | Полностью детерминирован |
| Парсинг | Нужно извлекать букву из текста | Сразу получаем букву |
| Скорость | Генерация нескольких токенов | Один forward pass + 4 мини-прохода |
| Калибровка | Сложно получить confidence | Logprobs напрямую = уверенность |

---

# CountryPriorReranker: Полное объяснение

## 1. Зачем нужен Reranker?

**Проблема**: В датасете MCQ варианты ответов часто специфичны для страны. Например:

| Вопрос | Вариант | Страна |
|--------|---------|--------|
| "Какой праздник отмечают 4 июля?" | "День независимости" | US |
| "Какой праздник отмечают 4 июля?" | "Обычный день" | UK, China, Iran |
| "Как называется глава государства?" | "President" | US, Iran |
| "Как называется глава государства?" | "Prime Minister" | UK |
| "Как называется глава государства?" | "Chairman" | China |

**Идея**: Если мы знаем, для какой страны задаётся вопрос, можно **добавить бонус** к вариантам, которые статистически чаще правильны для этой страны.

---

## 2. Построение априорного распределения (Prior)

### Шаг 1: Сбор статистики из тренировочных данных

```python
@classmethod
def from_train_csv(cls, train_csv_path, target_countries=("US", "UK", "China", "Iran"), alpha=1.0):
```

Метод читает `train_dataset_mcq.csv` и для каждого текста варианта считает, **сколько раз он встречался для каждой страны**:

```python
for row in r:
    choices = _parse_mapping_str(row.get("choices", ""))           # {"A": "President", "B": "King", ...}
    choice_countries = _parse_mapping_str(row.get("choice_countries", ""))  # {"A": "US", "B": "UK", ...}

    for letter, option_text in choices.items():
        tag = choice_countries.get(letter)  # Страна для этого варианта
        norm_text = _normalize_option_text(option_text)
        counts[norm_text][tag] += 1
```

**Результат** — таблица подсчётов:

```
counts = {
    "president":           {"US": 45, "Iran": 12, "UK": 2,  "China": 1},
    "prime minister":      {"US": 3,  "Iran": 1,  "UK": 38, "China": 2},
    "chairman":            {"US": 1,  "Iran": 0,  "UK": 1,  "China": 29},
    "independence day":    {"US": 52, "Iran": 5,  "UK": 3,  "China": 2},
    ...
}
```

### Шаг 2: Преобразование в log-вероятности с Laplace smoothing

```python
for text, tag_counts in counts.items():
    total = sum(tag_counts.values())
    denom = total + alpha * len(target_countries)  # Laplace smoothing

    for country in target_countries:
        p = (tag_counts.get(country, 0) + alpha) / denom
        per_country[country] = math.log(p)
```

**Формула (Laplace-smoothed probability):**

$$P(\text{country} | \text{option\_text}) = \frac{\text{count}(\text{option\_text}, \text{country}) + \alpha}{\sum_{c} \text{count}(\text{option\_text}, c) + \alpha \cdot |\text{countries}|}$$

где $\alpha = 1.0$ — параметр сглаживания (чтобы не было нулевых вероятностей).

**Пример для "president":**

```
counts = {"US": 45, "Iran": 12, "UK": 2, "China": 1}
total = 45 + 12 + 2 + 1 = 60
denom = 60 + 1.0 * 4 = 64

P(US | "president")    = (45 + 1) / 64 = 0.719  →  log = -0.33
P(Iran | "president")  = (12 + 1) / 64 = 0.203  →  log = -1.59
P(UK | "president")    = (2 + 1) / 64  = 0.047  →  log = -3.06
P(China | "president") = (1 + 1) / 64  = 0.031  →  log = -3.47
```

---

## 3. Вычисление бонуса (centered log-prior)

```python
def bonus(self, option_text: str, target_country: str) -> float:
    norm_text = _normalize_option_text(option_text)
    logp = self._text_logp_by_country.get(norm_text, {}).get(target_country)
    if logp is None:
        return 0.0
    # Center relative to uniform
    return logp - self._uniform_logp
```

**Формула бонуса:**

$$\text{bonus}(\text{option}, \text{country}) = \log P(\text{country} | \text{option}) - \log \frac{1}{|\text{countries}|}$$

$$= \log P(\text{country} | \text{option}) + \log |\text{countries}|$$

**Зачем центрирование?** Чтобы бонусы были **сравнимы между странами**:
- Бонус > 0 означает: "этот вариант чаще связан с этой страной, чем в среднем"
- Бонус < 0 означает: "этот вариант реже связан с этой страной"
- Бонус = 0 означает: "нет данных" или "равномерное распределение"

**Пример:**

```
uniform_logp = log(1/4) = -1.39

bonus("president", "US")    = -0.33 - (-1.39) = +1.06  ✓ Положительный!
bonus("president", "China") = -3.47 - (-1.39) = -2.08  ✗ Отрицательный
```

---

## 4. Интеграция в Logprob Scoring

В функции `choose_mcq_via_logprob`:

```python
for choice in cfg.choice_letters:
    # ... вычисляем logprob для варианта ...
    score = best_lp  # базовый logprob от модели

    if (reranker is not None
        and rerank_weight
        and target_country
        and mcq_choices is not None
        and choice in mcq_choices):

        score += float(rerank_weight) * reranker.bonus(mcq_choices[choice], target_country)

    scores[choice] = score
```

**Итоговая формула скоринга:**

$$\text{score}(c) = \underbrace{\log P(c | \text{prompt})}_{\text{от модели}} + \underbrace{w \cdot \text{bonus}(\text{option\_text}_c, \text{country})}_{\text{от reranker}}$$

где:
- $c \in \{A, B, C, D\}$ — буква варианта
- $\text{option\_text}_c$ — текст варианта (например, "President")
- $w$ = `rerank_weight` — вес влияния reranker'а

---

## 5. Визуализация полного пайплайна

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ВХОДНЫЕ ДАННЫЕ                                     │
│  prompt: "Вопрос для US: Как называется глава государства?"                     │
│  mcq_choices: {"A": "President", "B": "Prime Minister", "C": "King", "D": "Chairman"}
│  target_country: "US"                                                           │
│  rerank_weight: 0.5                                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
┌─────────────────────────────────────────────────────────┐       ┌─────────────────────────────────────┐
│        LOGPROB SCORING          │       │        COUNTRY PRIOR RERANKER       │
│         (от модели)             │       │         (из статистики)             │
│                                 │       │                                     │
│  logP("A" | prompt) = -0.80     │       │  bonus("President", "US") = +1.06   │
│  logP("B" | prompt) = -1.20     │       │  bonus("Prime Minister", "US")=-0.8 │
│  logP("C" | prompt) = -2.50     │       │  bonus("King", "US") = -1.2         │
│  logP("D" | prompt) = -1.50     │       │  bonus("Chairman", "US") = -2.1     │
└─────────────────────────────────────────────────────────┘       └─────────────────────────────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         КОМБИНИРОВАНИЕ SCORES                                   │
│                                                                                 │
│  score(A) = -0.80 + 0.5 × (+1.06) = -0.80 + 0.53 = -0.27  ← ЛУЧШИЙ             │
│  score(B) = -1.20 + 0.5 × (-0.80) = -1.20 - 0.40 = -1.60                       │
│  score(C) = -2.50 + 0.5 × (-1.20) = -2.50 - 0.60 = -3.10                       │
│  score(D) = -1.50 + 0.5 × (-2.10) = -1.50 - 1.05 = -2.55                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ВЫХОД: "A"                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Влияние rerank_weight

| `rerank_weight` | Поведение |
|-----------------|-----------|
| `0.0` | Reranker полностью выключен, только logprobs модели |
| `0.1 - 0.3` | Лёгкая коррекция, модель доминирует |
| `0.5 - 1.0` | Сбалансированное влияние |
| `> 1.0` | Prior доминирует над моделью (опасно, может overfit) |

**Математически:**

$$\text{score}(c) = \log P_{\text{model}}(c) + w \cdot \text{bonus}(c)$$

При $w \to \infty$ ответ определяется **только статистикой из тренировочных данных**, модель игнорируется.

---

## 7. Граничные случаи

```python
if logp is None:
    return 0.0  # Нет данных → нейтральный бонус
```

Если текст варианта **не встречался** в тренировочных данных:
- `bonus = 0.0`
- Reranker не влияет на этот вариант
- Решение принимается только на основе logprobs модели

---

## 8. Интуиция

**CountryPriorReranker** — это **Байесовский prior**:

$$P(\text{answer} | \text{question}, \text{country}) \propto \underbrace{P(\text{answer} | \text{question})}_{\text{likelihood от модели}} \times \underbrace{P(\text{answer} | \text{country})}_{\text{prior от reranker}}$$

В log-пространстве:

$$\log P(\text{answer}) = \log P_{\text{model}} + \log P_{\text{prior}}$$

Это классическая схема **Байесовского вывода**, где:
- **Likelihood** = что говорит модель на основе контекста вопроса
- **Prior** = что мы знаем заранее о связи вариантов со странами
