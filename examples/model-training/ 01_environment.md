# Часть 1: Настройка окружения и подготовка данных

На этом этапе мы развернем рабочее окружение, проверим доступные аппаратные ресурсы видеокарты, загрузим базовую модель и проведем токенизацию текстового датасета.

## Шаг 1: Проверка памяти GPU

Перед стартом ресурсоемких задач необходимо убедиться в доступности видеопамяти (VRAM), чтобы избежать ошибок нехватки памяти (Out of Memory).

```python
!nvidia-smi
```

### Результат выполнения:

```text
Thu Oct 22 12:52:23 2026
+-----------------------------------------------------------------------------------------+

| NVIDIA-SMI 595.91.07              Driver Version: 595.91.07      CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+

| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       On  |   00000000:00:1E.0 Off |                    0 |
| N/A   33C    P0             33W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

> 💡 **Техническая справка:** `nvidia-smi` — системная утилита NVIDIA для мониторинга видеокарт. Она показывает текущую модель GPU (серверная Tesla T4 на 15 ГБ), рабочую температуру, энергопотребление и точный объем занятой памяти VRAM.

---

## Шаг 2: Импорт библиотек

Установим внешние пакеты и импортируем необходимые компоненты экосистемы Hugging Face и PyTorch.

```python
%%capture
!pip install -r requirements.txt
```

```python
%%capture

import warnings
warnings.filterwarnings("ignore")  # Игнорировать предупреждения

import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
from datasets import Dataset, load_dataset, disable_caching
disable_caching() ## Отключение кэширования для экономии диска

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextDataset

import torch
from torch.utils.data import Dataset, random_split
from transformers import TrainingArguments, Trainer
import accelerate
import bitsandbytes

from IPython.display import Markdown
```

### 💡 Краткий разбор ключевых библиотек:

- **`AutoModelForCausalLM`** — автоматически загружает архитектуру для каузального языкового моделирования (генерация текста токен за токеном слева направо, как в семействе GPT).
- **`bitsandbytes`** — библиотека для динамического 8-битного и 4-битного квантования весов, снижающая аппаратные требования к памяти GPU.
- **`accelerate`** — утилита, оптимизирующая распределение вычислений на железе без ручной настройки конфигурации в PyTorch.

---

## Шаг 3: Подготовка датасета для обучения

В качестве обучающей выборки используется файл **Amazon SageMaker FAQs**, содержащий пары вопрос-ответ в колонках `instruction` and `response`.

```python
sagemaker_faqs_dataset = load_dataset("csv", data_files='data/amazon_sagemaker_faqs.csv')['train']
sagemaker_faqs_dataset
```

### Результат выполнения:

```text
Dataset({
    features: ['instruction', 'response'],
    num_rows: 154
})
```

Пример сырой структуры данных:

```python
{'instruction': 'What is Amazon SageMaker?',
 'response': 'Amazon SageMaker is a fully managed service to prepare data and build, train, and deploy machine learning (ML) models...'}
```

### Шаг 3.1: Применение шаблона инструкций (Prompt Engineering)

Чтобы модель понимала, где заканчивается вопрос и начинается ответ, текст оборачивается в структурированный шаблон `PROMPT`:

```python
from utils.helpers import INTRO_BLURB, INSTRUCTION_KEY, RESPONSE_KEY, END_KEY, RESPONSE_KEY_NL, DEFAULT_SEED, PROMPT
Markdown(PROMPT)
```

#### Вид шаблона:

> Below is an instruction that describes a task. Write a response that appropriately completes the request.
>
> ### Instruction:
>
> {instruction}
>
> ### Response:
>
> {response}
>
> ### End

Напишем функцию форматирования и применим её ко всему датасету методом `.map()`:

```python
def _add_text(rec):
    instruction = rec["instruction"]
    response = rec["response"]
    if not instruction or not response:
        raise ValueError(f"Fields cannot be null in record: {rec}")
    rec["text"] = PROMPT.format(instruction=instruction, response=response)
    return rec

sagemaker_faqs_dataset = sagemaker_faqs_dataset.map(_add_text)
```
