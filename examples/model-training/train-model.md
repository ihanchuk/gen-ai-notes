### Дообучаем модель с прмименением АВС Bedrock, Sagemaker и JypeterNotebook ### 

#### Step 1 - Проверяем видео карту и гпу ####

```shell
!nvidia-smi
```

```nvidia-smi``` — утилита NVIDIA, которая показывает состояние видеокарты:
- модель GPU;
- объём видеопамяти и её использование;
- загрузку GPU;
- температуру и энергопотребление;
- процессы, использующие видеокарту;
- версию драйвера и поддерживаемую драйвером версию CUDA.

```!``` восклицательный знак означает «выполни команду оболочки».

#### Step 2 - Ставим зависиммости ####

```shell
%%capture
!pip install -r requirements.txt
```

```%%capture``` — магическая команда Jupyter, которая перехватывает и скрывает стандартный вывод и ошибки ячейки.

#### Step 3 - Начнем с импорта либок ####

```python
%%capture

import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
from datasets import Dataset, load_dataset, disable_caching
disable_caching() ## disable huggingface cache

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

#### Step 4 - Разбираем CSV на датасет ####

```python
sagemaker_faqs_dataset = load_dataset("csv", data_files='data/amazon_sagemaker_faqs.csv')['train']
sagemaker_faqs_dataset
```

> [!WARNING]
> Файл .csv ___Должен___ содержать таблицу с из 2 столбцов instruction и response. Это наши инструкции модели и примеры ответов.

В ответ на этот шаг Юпитер нам выдаст что-то вроде

```bash
Dataset({
    features: ['instruction', 'response'],
    num_rows: 154
})
```

##### проверим данные ##### 

```python
sagemaker_faqs_dataset[0]
```

в ответ мы ожидаем увидеть что-то типа

```JSON
{
   "instruction":"What is Amazon SageMaker?",
   "response":"Amazon SageMaker is ... bla-bla-bla"
}
```

#### Step 4.1 - Готовим тестовые промпты ####

```python
from utils.helpers import INTRO_BLURB, INSTRUCTION_KEY, RESPONSE_KEY, END_KEY, RESPONSE_KEY_NL, DEFAULT_SEED, PROMPT
'''
PROMPT = """{intro}
            {instruction_key}
            {instruction}
            {response_key}
            {response}
            {end_key}"""
'''
Markdown(PROMPT)
```

Добавим хелпер функции для обработки рекордов

```python
def _add_text(rec):
    instruction = rec["instruction"]
    response = rec["response"]

    if not instruction:
        raise ValueError(f"Expected an instruction in: {rec}")

    if not response:
        raise ValueError(f"Expected a response in: {rec}")

    rec["text"] = PROMPT.format(
        instruction=instruction, response=response)

    return rec
```

И применим хелпер для обработки датасета

```
sagemaker_faqs_dataset = sagemaker_faqs_dataset.map(_add_text)
sagemaker_faqs_dataset[0]
```

В ответ мы должны получить что-то типа

```json
{
   "instruction":"What is Amazon SageMaker?",
   "response":"Amazon SageMaker is a fully managed service to prepare data and build, train, and deploy machine learning (ML) models for any use case with fully managed infrastructure, tools, and workflows.",
   "text":"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n            ### Instruction:\n            What is Amazon SageMaker?\n            ### Response:\n            Amazon SageMaker is a fully managed service to prepare data and build, train, and deploy machine learning (ML) models for any use case with fully managed infrastructure, tools, and workflows.\n            ### End"
}
```

#### Step 5 - Подгружаем натрененную модель ####

Чтобы загрузить предварительно обученную модель, инициализируем токенизатора и базовую модель databricks/dolly-v2-3b. Веса модели хранятся в бакете Amazon S3. Первая ячейка кода загружает их в экземпляр ноутбука (это занимает около 2 минут). Затем библиотека Hugging Face Transformers загружает модель.

Токенизатор преобразует исходный текст в токены, а базовая модель генерирует текст на основе заданного промпта. Следуя приведённым ранее инструкциям, мы сможем правильно создать эти компоненты и использовать их в коде.

Для создания экземпляра токенизатора используется функция Python AutoTokenizer.from_pretrained().
- ```padding_side="left"``` указывает, с какой стороны последовательностей добавляются токены заполнения (padding). В данном случае они добавляются слева от каждой последовательности.
- ```eos_token``` — специальный токен, обозначающий конец последовательности. Присваивая его значение параметру pad_token, мы указываем, что для заполнения последовательностей также используется токен конца последовательности. Это может быть полезно при генерации текста, поскольку модель может учитывать этот токен как признак завершения генерации.
- ```tokenizer.add_special_tokens...``` добавляет три дополнительных специальных токена в словарь токенизатора. Вероятно, эти токены выполняют определённые функции в приложении, использующем токенизатор. Например, они могут обозначать конец входных данных, инструкции или ответа в диалоговой системе.
После выполнения функции объект tokenizer будет инициализирован и готов к токенизации текста.

#### Разберемся детальнее с терминами ####

___Токен___ — единица текста, с которой работает языковая модель. Это может быть целое слово, часть слова, знак препинания или другой фрагмент текста.

Например, слово unbelievable может разбиваться на несколько токенов. Модель получает текст в виде последовательности токенов, представленных числовыми ID.

___Токенизатор___ — компонент, который преобразует обычный текст в токены и затем может преобразовать токены обратно в текст.

Например:
```text
"Hello world!" → [15496, 995, 0]

Это условный пример: реальные ID зависят от токенизатора.
```

__pad_token___ — специальный токен-заполнитель. Он используется, когда последовательности разной длины нужно выровнять до одинаковой длины, например для обработки нескольких текстов одним batch.

```text
Пример:
Hello world [PAD] [PAD]
Hi          [PAD] [PAD] [PAD]
```


___eos_token (End Of Sequence)___ — специальный токен, обозначающий конец последовательности.

Он может сигнализировать модели, что генерация ответа завершена. В некоторых конфигурациях eos_token также используется как pad_token, хотя их функции различаются

___padding_side___ — настройка токенизатора, определяющая, с какой стороны добавлять токены-заполнители.
- padding_side="left" — слева.
- padding_side="right" — справа.

```python
import os
import boto3

# Download the pre-trained Dolly v2 3B model from the lab's S3 bucket (about 2 minutes).
BASE_MODEL_DIR = "dolly-v2-3b"
BASE_MODEL_PREFIX = "models/dolly-v2-3b/"

if not os.path.exists(os.path.join(BASE_MODEL_DIR, "pytorch_model.bin")):
    s3_client = boto3.client("s3")
    code_bucket = next(
        b["Name"] for b in s3_client.list_buckets()["Buckets"] if b["Name"].startswith("lab-code")
    )
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=code_bucket, Prefix=BASE_MODEL_PREFIX):
        for obj in page.get("Contents", []):
            file_name = obj["Key"][len(BASE_MODEL_PREFIX):]
            if file_name:
                print(f"Downloading {file_name}")
                s3_client.download_file(code_bucket, obj["Key"], os.path.join(BASE_MODEL_DIR, file_name))

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR,
                                          padding_side="left")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens":
                              [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})
```

```python
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    # use_cache=False,
    device_map="auto", #"balanced",
    load_in_8bit=True,
)
```

#### Step 6 - Готовим модель для тренировки ####

Перед обучением INT8-модели с использованием PEFT (Parameter-Efficient Fine-Tuning, параметрически эффективного дообучения) требуется выполнить предварительную обработку. Для этого импортируется вспомогательная функция prepare_model_for_int8_training, которая выполняет следующие действия:
- Преобразует все модули, не использующие INT8, в формат полной точности FP32 для повышения стабильности обучения.
- Добавляет forward_hook к слою входных эмбеддингов, чтобы обеспечить вычисление градиентов для входных скрытых состояний.
- Включает gradient checkpointing для более эффективного использования памяти при обучении.

```python
model.resize_token_embeddings(len(tokenizer))
```

Используйте функцию ```preprocess_batch``` для предварительной обработки текстового поля пакета данных, выполняя токенизацию, усечение и другие необходимые операции в соответствии с заданной максимальной длиной.

Функция принимает на вход пакет данных, токенизатор и максимальную длину последовательности.

Подробнее см. в файле mlu_utils/helpers.py.

```python
from functools import partial
from utils.helpers import mlu_preprocess_batch

MAX_LENGTH = 256
_preprocessing_function = partial(mlu_preprocess_batch, max_length=MAX_LENGTH, tokenizer=tokenizer)
```

Затем примените функцию предварительной обработки к каждому пакету данных в датасете, соответствующим образом изменяя текстовое поле. Операция map выполняется пакетами, при этом из датасета удаляются столбцы instruction, response и text.

В результате создаётся processed_dataset путём фильтрации sagemaker_faqs_dataset по длине поля input_ids, чтобы убедиться, что она не превышает заданное значение MAX_LENGTH

```python
encoded_sagemaker_faqs_dataset = sagemaker_faqs_dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "response", "text"],
)

processed_dataset = encoded_sagemaker_faqs_dataset.filter(lambda rec: len(rec["input_ids"]) < MAX_LENGTH)
```
Разделите датасет на две части — train (для обучения) и test (для оценки модели).

```python
split_dataset = processed_dataset.train_test_split(test_size=14, seed=0)
split_dataset
```

### Step 7. Определение Trainer и дообучение LLM ###
Для эффективного дообучения модели в этой практической лабораторной работе используется ```LoRA (Low-Rank Adaptation).```

```LoRA``` замораживает веса предварительно обученной модели и добавляет в каждый слой архитектуры Transformer обучаемые матрицы низкорангового разложения. Это значительно сокращает количество параметров, которые необходимо обучать для решения последующих задач.

По сравнению с дообучением GPT-3 с 175 млрд параметров с использованием оптимизатора ```Adam```, ```LoRA ``` позволяет сократить количество обучаемых параметров в 10 000 раз, а потребность в видеопамяти GPU — в 3 раза.
Шаг 5.1. Определение LoraConfig и загрузка модели LoRA
Используйте класс ```LoraConfig``` из библиотеки ```Hugging Face PEFT```, предназначенной для параметрически эффективного дообучения современных моделей.

В ```LoraConfig``` укажите следующие параметры:
- ```r``` — размерность низкоранговых матриц.
- ```lora_alpha``` — коэффициент масштабирования низкоранговых матриц.
- ```lora_dropout``` — вероятность применения dropout в слоях LoRA.

