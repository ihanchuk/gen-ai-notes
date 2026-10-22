# Часть 2: Тонкая настройка модели LoRA

На этом этапе мы выполним загрузку весов базовой языковой модели, проведем токенизацию подготовленного датасета и настроим процесс эффективного дообучения с помощью LoRA.

## Шаг 4: Загрузка предобученной LLM и токенизация

Скачаем веса модели **Dolly v2 3B** из облачного хранилища S3 лабораторной среды и инициализируем токенизатор.

```python
import os
import boto3

BASE_MODEL_DIR = "dolly-v2-3b"
BASE_MODEL_PREFIX = "models/dolly-v2-3b/"

if not os.path.exists(os.path.join(BASE_MODEL_DIR, "pytorch_model.bin")):
    s3_client = boto3.client("s3")
    code_bucket = next(b["Name"] for b in s3_client.list_buckets()["Buckets"] if b["Name"].startswith("lab-code"))
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=code_bucket, Prefix=BASE_MODEL_PREFIX):
        for obj in page.get("Contents", []):
            file_name = obj["Key"][len(BASE_MODEL_PREFIX):]
            if file_name:
                s3_client.download_file(code_bucket, obj["Key"], os.path.join(BASE_MODEL_DIR, file_name))

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})
```

> 🔍 **Техническая деталь:** Параметр `padding_side="left"` критически важен для генеративных моделей (Causal LM). Добавление заполняющих токенов (паддинга) слева гарантирует, что значащие токены промпта окажутся с правой стороны, позволяя модели корректно выстраивать контекст предсказания следующего слова.

Загрузим саму модель в режиме квантования INT8 для минимизации расхода памяти:

```python
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    device_map="auto",
    load_in_8bit=True,
)
model.resize_token_embeddings(len(tokenizer))
```

### Предобработка и очистка батчей

Зададим максимальную длину контекста в 256 токенов, векторизуем текст и разобьем выборку на обучение и валидацию.

```python
from functools import partial
from utils.helpers import mlu_preprocess_batch

MAX_LENGTH = 256
_preprocessing_function = partial(mlu_preprocess_batch, max_length=MAX_LENGTH, tokenizer=tokenizer)

encoded_sagemaker_faqs_dataset = sagemaker_faqs_dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "response", "text"],
)

processed_dataset = encoded_sagemaker_faqs_dataset.filter(lambda rec: len(rec["input_ids"]) < MAX_LENGTH)
split_dataset = processed_dataset.train_test_split(test_size=14, seed=0)
```

---

## Шаг 5: Конфигурация LoRA и обучение

Применим метод низкоранговой адаптации **LoRA**, чтобы настроить только малую часть весов, зафиксировав базовую модель.

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

# Подготовка к 8-битному обучению и заморозка базовых весов
model = prepare_model_for_int8_training(model)
for param in model.parameters():
    if param.dtype in [torch.float32, torch.float16]:
        param.requires_grad = False

# Разморозка выходных слоев для точности вычислений
num_layers_to_unfreeze = 2
for layer in model.gpt_neox.layers[-num_layers_to_unfreeze:]):
    for param in layer.parameters():
        if param.dtype in [torch.float32, torch.float16]:
            param.requires_grad = True

# Конфигурация гиперпараметров LoRA
lora_config = LoraConfig(
                 r=256,              # Ранг матриц декомпозиции
                 lora_alpha=512,      # Множитель масштабирования весов LoRA
                 lora_dropout=0.05,   # Коэффициент регуляризации (дропаут)
                 bias="none",
                 task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### Инициализация тренера и запуск обучения

Используем специализированный дата-коллатор `MLUDataCollatorForCompletionOnlyLM` для маскирования лосса на вопросах и зададим параметры сессии обучения.

```python
from utils.helpers import MLUDataCollatorForCompletionOnlyLM

data_collator = MLUDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
)

training_args = TrainingArguments(
                    output_dir="dolly-3b-lora",
                    fp16=True,
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    learning_rate=1e-4,
                    num_train_epochs=10,
                    logging_strategy="steps",
                    logging_steps=100,
                    evaluation_strategy="steps",
                    eval_steps=100,
                    save_strategy="steps",
                    save_total_limit=10,
                    optim="adamw_torch"
)

trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
)
model.config.use_cache = False
trainer.train()
```

### Сохранение результатов

Сохраним только веса адаптера (что делает артефакт компактным) и файлы конфигурации токенизатора:

```python
trainer.model.save_pretrained("dolly-3b-lora")
trainer.save_model()
trainer.model.config.save_pretrained("dolly-3b-lora")
tokenizer.save_pretrained("dolly-3b-lora")
```
