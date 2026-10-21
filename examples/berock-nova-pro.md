```shell
!pip install -U boto3 botocore --quiet
```
```python
import boto3, json

# Amazon Bedrock is fully managed and serverless: there is no endpoint to deploy or
# instance to manage. We call the Amazon Nova Pro foundation model on demand through the
# Bedrock Runtime "Converse" API.
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Amazon Nova Pro is served through a cross-Region inference profile in Amazon Bedrock.
MODEL_ID = "us.amazon.nova-pro-v1:0"

print(f"boto3 {boto3.__version__} ready. Model: {MODEL_ID}")
```

```python
# Uses the "bedrock" client and "MODEL_ID" created in the setup cell above.

def query_model(prompt, temperature=0.7, max_tokens=1024, system=None):
    """Send a prompt to Amazon Nova Pro through the Amazon Bedrock Converse API.

    Args:
        prompt (str): The text prompt to send to the model.
        temperature (float): Controls randomness (0.0 to ~1.0). Lower is more focused.
        max_tokens (int): Maximum number of tokens the model should generate.
        system (str): Optional system instruction that sets the model behavior.

    Returns:
        str: The generated text.
    """
    kwargs = {
        "modelId": MODEL_ID,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        kwargs["system"] = [{"text": system}]

    response = bedrock.converse(**kwargs)

    # The assistant reply is a list of content blocks; collect the text blocks.
    content = response["output"]["message"]["content"]
    text = "".join(block.get("text", "") for block in content)

    usage = response.get("usage", {})
    print(text)
    print(f"\n[tokens: input={usage.get('inputTokens')}, output={usage.get('outputTokens')}]")
    return text
```

```python
code_prompt = "Write a Python function that returns the factorial of a number."
_ = query_model(code_prompt, temperature=0.2)
```

```python
prompt = "What is AWS Lambda?"
print("--- Low temperature (focused, temperature=0.2) ---")
_ = query_model(prompt, temperature=0.2, max_tokens=512)
```
