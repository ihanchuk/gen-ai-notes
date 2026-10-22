# Часть 3: Деплой инференса в Amazon SageMaker

После успешного обучения нам необходимо упаковать веса LoRA-адаптера, передать их в облачное хранилище S3 и развернуть постоянный HTTP-эндпоинт инференса в Amazon SageMaker с использованием Deep Java Library (DJL).

## Шаг 6: Развертывание модели в SageMaker

### Шаг 6.1: Инициализация окружения AWS

Создадим рабочую сессию, определим IAM-роль инстанса и подтянем официальный Docker-контейнер `djl-deepspeed`, оптимизированный для инференса крупных LLM.

```python
%%capture
!pip3 install sagemaker==2.237.1
```

```python
import boto3
import json
import sagemaker.djl_inference
from sagemaker.session import Session
from sagemaker import image_uris
from sagemaker import Model

sagemaker_session = Session()
aws_role = sagemaker_session.get_caller_identity_arn()
aws_region = boto3.Session().region_name

image_uri = image_uris.retrieve(framework="djl-deepspeed",
                                version="0.22.1",
                                region=sagemaker_session._region_name)
print("Docker Image URI для DJL DeepSpeed: ", image_uri)
```

---

### Шаг 6.2: Формирование артефакта модели

Сервер DJL требует определенной структуры файлов внутри архива. Создадим структуру папок и перенесем туда обученные веса адаптера. Базовая модель будет скачана контейнером с Hugging Face Hub автоматически при запуске.

```bash
%%bash
rm -rf lora_model
mkdir -p lora_model/dolly-3b-lora
cp dolly-3b-lora/adapter_config.json lora_model/dolly-3b-lora/
cp dolly-3b-lora/adapter_model.bin lora_model/dolly-3b-lora/
```

Создадим конфигурационный файл `serving.properties`, описывающий параметры инференс-сервера:

```python
%%writefile lora_model/serving.properties
engine=Python
option.entryPoint=model.py
option.adapter_checkpoint=dolly-3b-lora
option.adapter_name=dolly-lora
option.model_loading_timeout=1200
```

Запишем требования к пакетному окружению контейнера в файл `requirements.txt`:

```python
%%writefile lora_model/requirements.txt
accelerate>=0.16.0,<1
bitsandbytes==0.39.0
boto3
click>=8.0.4,<9
datasets>=2.10.0,<3
deepspeed>=0.8.3,<0.9
faiss-cpu==1.7.4
ipykernel==6.22.0
scipy==1.11.1
torch>=2.0.0
transformers==4.28.1
peft==0.3.0
pytest==7.3.2
```

Перенесем исполняемый Python-скрипт инференса (который содержит описание пайплайна генерации текста) в корень сборки:

```bash
%%bash
cp utils/deployment_model.py lora_model/model.py
```

---

### Шаг 6.3: Архивация и отправка на AWS S3

Скомпилируем папку проекта в архив формата `.tar.gz` и отправка файла в бакет S3.

```bash
%%bash
tar -cvzf lora_model.tar.gz lora_model/
```

```python
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

# Поиск целевого бакета инфраструктуры по префиксу названия
for bucket in s3.buckets.all():
    if bucket.name.startswith('artifact'):
        mybucket = bucket.name
        print("Целевой бакет найден:", mybucket)

s3_client.upload_file("lora_model.tar.gz", mybucket, "lora_model.tar.gz")
```

---

### Шаг 6.4: Создание конфигурации эндпоинта и деплой

Зададим сетевые параметры модели и отправим запрос на создание выделенного инференс-сервера на базе инстанса GPU `ml.g4dn.2xlarge`.

```python
model_data = "s3://{}/lora_model.tar.gz".format(mybucket)

model = Model(image_uri=image_uri,
              model_data=model_data,
              predictor_cls=sagemaker.djl_inference.DJLPredictor,
              role=aws_role)
```

```python
%%time
# Инициализация деплоя без блокировки потока выполнения (wait=False)
predictor = model.deploy(1, "ml.g4dn.2xlarge", wait=False)
```

Развертывание виртуальной машины и инициализация весов в памяти GPU занимает около 5–10 минут. Напишем блокирующий цикл для отслеживания изменения статуса сервера:

```python
import time
sm_client = boto3.client('sagemaker')

while True:
    status = sm_client.describe_endpoint(EndpointName=predictor.endpoint_name)['EndpointStatus']
    print(f"Текущий статус эндпоинта: {status}")
    if status == 'InService':
        print("🚀 Эндпоинт успешно запущен и готов принимать запросы!")
        break
    time.sleep(30)
```

---

## Шаг 7: Тестирование развернутой модели (Inference)

Теперь проверим качество генерации ответов нашей дообученной модели, отправив проверочный запрос в формате JSON.

```python
# Отправка тестового вопроса на удаленный сервер инференса
outputs = predictor.predict({"inputs": "What security measures does Amazon SageMaker have?"})
```

Выведем текстовый ответ, присланный моделью, на экран с помощью Markdown-разметки:

```python
from IPython.display import Markdown
Markdown(outputs)
```
