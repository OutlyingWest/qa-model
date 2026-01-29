## Personal info
name: Aleksei Buvailik
codabench nickname: albu670g
matriculation number: 5271683
albu670g@mailbox.tu-dresden.de

## Important to mention
- Как была имплементирована LoRA - для каких слоев тренировался адаптер
- Как был имплементирован logprob
- Как был имплементирован RAG

## Last RAG experiment
### Last RAG Training with adding stop‑words/stammer features for indexing
Вижу стабильный прогресс по метрикам и плавное насыщение качества; улучшения от stop‑words/stammer выглядят умеренными, без резких скачков.

- Loss падает уверенно до ~1.55 к 1.5 эпохе и дальше выходит на плато 1.46–1.52 к 3 эпохе
- Eval: eval_loss 1.545 → 1.466, eval_mean_token_accuracy 0.671 → 0.680 — прирост есть, но уже небольшой.
- Энтропия снижается (1.46–1.42), модель становится увереннее; при этом mean_token_accuracy почти не растет после ~2 эпох.

Что можно сказать: тренд хороший, нужна проверка качества
генераций и/или данных

#### Inference results
Не удалось достичь сколько нибудь значимого улучшения с текущей имплементацией RAG

MCQ Results
Metric	Score
Overall
Accuracy	0.79
By Country
China	0.74
Iran	0.68
UK	0.91
US	0.84
SAQ Results
Metric	Score
Overall
Accuracy	0.58
By Country
CN	0.51
GB	0.63
IR	0.50
US	0.70

Написать направления для дальнейших возможных улучшений RAG

