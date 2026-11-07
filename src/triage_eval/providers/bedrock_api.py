import json
import os

from .common import normalize_chat, retry


def complete(
    configuration, prompt, max_tokens, temperature=None, seed=None, phase="solver"
):
    model_id = configuration["model_id"]
    if model_id.startswith(("deepseek.", "moonshotai.")):
        import boto3

        client = boto3.client(
            "bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens),
        }
        if temperature is not None:
            body["temperature"] = temperature
        if configuration.get("reasoning_effort"):
            body["reasoning_effort"] = configuration["reasoning_effort"]
        response = retry(
            lambda: client.invoke_model(
                modelId=model_id, body=json.dumps(body), contentType="application/json"
            )
        )
        return normalize_chat(json.loads(response["body"].read()), "bedrock", model_id)
    from litellm import completion

    options = {
        "model": f"bedrock/{model_id}",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "aws_region_name": os.environ.get("AWS_REGION", "us-east-1"),
    }
    if temperature is not None:
        options["temperature"] = temperature
    if configuration.get("reasoning_effort"):
        options["reasoning_effort"] = configuration["reasoning_effort"]
    if configuration.get("thinking"):
        options["thinking"] = configuration["thinking"]
    try:
        response = retry(lambda: completion(**options))
    except Exception as error:
        status = getattr(error, "status_code", None)
        if status != 400 or options.get("thinking", {}).get("type") != "adaptive":
            raise
        options["thinking"] = {
            "type": "enabled",
            "budget_tokens": max(1024, int(max_tokens) - 1024),
        }
        response = retry(lambda: completion(**options))
    return normalize_chat(response, "bedrock", model_id)
