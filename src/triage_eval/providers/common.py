import os
import re
import time


def require_env(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name} before using this provider")
    return value


def retry(function, attempts=20):
    for attempt in range(attempts):
        try:
            return function()
        except Exception as error:
            status = getattr(error, "status_code", None) or getattr(error, "code", None)
            response = getattr(error, "response", {})
            if isinstance(response, dict):
                status = (
                    response.get("ResponseMetadata", {}).get("HTTPStatusCode") or status
                )
            transient = status in {408, 409, 429, 500, 502, 503, 504}
            transient = transient or isinstance(error, (TimeoutError, ConnectionError))
            transient = transient or any(
                term in type(error).__name__.lower()
                for term in ("timeout", "ratelimit", "connection")
            )
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(min(2**attempt, 60))


def split_reasoning(text):
    if "</think>" in text:
        reasoning, answer = text.split("</think>", 1)
        return reasoning.replace("<think>", "").strip(), answer.strip()
    for marker in ("<|channel|>final<|message|>", "<|channel|>\nfinal<|message|>"):
        if marker in text:
            reasoning, answer = text.split(marker, 1)
            reasoning = re.sub(r"<\|[^>]+\|>", "", reasoning)
            answer = re.sub(r"<\|[^>]+\|>", "", answer)
            return reasoning.strip(), answer.strip()
    text = re.sub(r"<\|[^>]+\|>", "", text)
    return "", text.strip()


def normalize_chat(response, provider, model_id):
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    choice = response["choices"][0]
    message = choice.get("message", {})
    raw = message.get("content") or ""
    if isinstance(raw, list):
        raw = "".join(part.get("text", "") for part in raw if isinstance(part, dict))
    reasoning, text = split_reasoning(raw)
    reasoning = (
        message.get("reasoning_content") or message.get("reasoning") or reasoning
    )
    usage = response.get("usage") or {}
    return {
        "text": text,
        "raw_text": raw,
        "reasoning": reasoning or "",
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "usage": usage,
        "model_id": response.get("model") or model_id,
        "provider": provider,
        "finish_reason": choice.get("finish_reason"),
    }


def compatible_completion(
    configuration,
    prompt,
    max_tokens,
    temperature,
    base_url=None,
    seed=None,
    phase="solver",
):
    from openai import OpenAI

    provider = configuration["provider"]
    client = OpenAI(
        api_key=require_env(configuration["api_key_env"]),
        base_url=base_url,
        max_retries=0,
    )
    kwargs = {
        "model": configuration["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        configuration.get("max_tokens_parameter", "max_tokens"): int(max_tokens),
    }
    if temperature is not None and configuration.get("supports_temperature", True):
        kwargs["temperature"] = temperature
    effort = configuration.get("reasoning_effort")
    if effort:
        if provider == "openrouter":
            kwargs["extra_body"] = {"reasoning": {"effort": effort}}
        else:
            kwargs["reasoning_effort"] = effort
    if configuration.get("disable_reasoning") and phase == "planner":
        kwargs["extra_body"] = {"reasoning": {"enabled": False}}
    if configuration.get("extra_body"):
        kwargs.setdefault("extra_body", {}).update(configuration["extra_body"])
    response = retry(lambda: client.chat.completions.create(**kwargs))
    return normalize_chat(response, provider, configuration["model_id"])
