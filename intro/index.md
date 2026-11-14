# Ресурсы для изучения Machine Learning

Тебе нужен не курс по AWS и не курс по использованию готовых LLM. Нужна база **Machine Learning: как устроено обучение моделей, как выбирать алгоритмы, оценивать качество и улучшать результат**.

## Подборка ресурсов

### 1. Kaggle Learn — Intro to Machine Learning

- **Стоимость:** бесплатно.
- **Продолжительность:** около 3 часов.
- **Ссылка:** https://www.kaggle.com/learn/intro-to-machine-learning

Я бы начал именно с него.

Что изучишь:
- как работают ML-модели;
- подготовка данных;
- первая модель;
- Model Validation;
- underfitting и overfitting;
- Random Forest.

Главное преимущество: короткие практические упражнения прямо в браузере. Не нужно настраивать окружение.

### 2. Google Machine Learning Crash Course

- **Стоимость:** бесплатно.
- **Ссылка:** https://developers.google.com/machine-learning/crash-course

Это основной кандидат для изучения теории.

Особенно рекомендую следующие разделы:

| Тема | Что поймёшь |
|---|---|
| Linear Regression | Как модель учится предсказывать значения |
| Loss | Как измеряется ошибка |
| Gradient Descent | Как модель корректирует weights |
| Logistic Regression | Как работает классификация |
| Classification | Как оценивать предсказания |
| Generalization | Почему модель может плохо работать на новых данных |
| Overfitting | Почему модель переобучается |
| Neural Networks | Основы нейросетей |
| Embeddings | Как текст превращается в векторы |

Есть интерактивные визуализации, программные упражнения и тесты.

### 3. Machine Learning Specialization — Andrew Ng

- **Стоимость:** условия оплаты нужно проверить на Coursera; возможны варианты бесплатного доступа к отдельным материалам.
- **Ссылка:** https://www.coursera.org/specializations/machine-learning-introduction

Более систематический академический курс. Три части:

1. Supervised Machine Learning: Regression and Classification.
2. Advanced Learning Algorithms.
3. Unsupervised Learning, Recommenders, Reinforcement Learning.

Будешь работать с NumPy, scikit-learn и TensorFlow.

Это хороший вариант, если захочешь глубже разобраться в математике и алгоритмах. Перед записью проверь условия бесплатного доступа и стоимость полного обучения.

### 4. fast.ai — Practical Deep Learning for Coders

- **Стоимость:** бесплатно.
- **Ссылка:** https://course.fast.ai/

Это уже следующий уровень.

Здесь будешь самостоятельно создавать и обучать модели:
- Computer Vision;
- NLP;
- Tabular Data;
- Recommendation Systems;
- Random Forests;
- PyTorch и Hugging Face.

Курс рассчитан на людей с опытом программирования.

## Рекомендуемый порядок

1. **Kaggle Intro to ML** — получить первое практическое представление об обучении моделей.
2. **Google ML Crash Course** — разобраться в математике и механизмах обучения.
3. **Andrew Ng** — систематизировать знания и глубже понять алгоритмы.
4. **fast.ai** — перейти к полноценному Deep Learning и PyTorch.

## Что пока не стоит изучать

Не начинал бы сейчас с:
- SageMaker;
- Bedrock;
- LangChain;
- RAG;
- AgentCore;
- Fine-tuning LLM.

Это инструменты и прикладные направления. Без понимания базового ML будет сложнее разобраться, что именно происходит внутри обучения.

## Итоговая рекомендация

**Kaggle → Google ML Crash Course → fast.ai.**

Google ML Crash Course стоит проходить внимательно, а не просто ради галочки. Именно там разбираются `loss`, `weights`, `gradient descent`, `epochs` и `overfitting`. Это фундамент, который пригодится и при обучении классических моделей, и при fine-tuning LLM.

Andrew Ng можно добавить между Google и fast.ai, если захочется более систематического и глубокого курса.
