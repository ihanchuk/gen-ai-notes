```python
# Import necessary libraries
import json
import logging
import os

import boto3

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Amazon Bedrock runtime client. The Region is taken from the Lambda execution environment.
bedrock_runtime = boto3.client("bedrock-runtime")

# The Amazon Bedrock model the sample application calls. The lab ships a default model;
# the DIY has the learner point the application at a different model by changing this
# environment variable on the function configuration.
DEFAULT_MODEL_ID = "us.amazon.nova-pro-v1:0"

# CORS headers so the browser-based sample application can call the API.
CORS_HEADERS = {
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "OPTIONS,POST",
}


def _response(payload):
    # The sample application expects a list of objects, each with a generated_text field.
    return {
        "statusCode": 200,
        "headers": CORS_HEADERS,
        "body": json.dumps(payload),
    }


def lambda_handler(event, context):
    # Log the incoming event in JSON format
    logger.info("Event: %s", json.dumps(event))

    # The sample app posts {"inputs": "<prompt>", "parameters": {"max_new_tokens": 400}}
    raw_body = event.get("body") or "{}"
    if isinstance(raw_body, str):
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            body = {"inputs": raw_body}
    else:
        body = raw_body

    prompt = (body.get("inputs") or "").strip()
    parameters = body.get("parameters") or {}
    max_tokens = int(parameters.get("max_new_tokens", 512))

    # The model id is read from the environment so the application's model can be changed
    # without editing code (this is the lever the DIY task uses).
    model_id = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
    logger.info("Invoking Amazon Bedrock model: %s", model_id)

    try:
        result = bedrock_runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.5},
        )
        # Return only the assistant text. Some models also emit a reasoningContent block,
        # which is skipped so the application shows just the answer.
        content = result["output"]["message"]["content"]
        generated_text = "".join(part["text"] for part in content if "text" in part).strip()
    except Exception as error:  # surface a readable message to the sample app
        logger.exception("Amazon Bedrock invocation failed")
        generated_text = f"Error generating a response: {error}"

    return _response([{"generated_text": generated_text}])

```
