import os

from .common import require_env, retry


def complete(
    configuration, prompt, max_tokens, temperature=None, seed=None, phase="solver"
):
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or require_env("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    options = {
        "max_output_tokens": int(max_tokens),
        "thinking_config": types.ThinkingConfig(
            thinking_budget=configuration.get("thinking_budget", 0)
        ),
    }
    if temperature is not None:
        options["temperature"] = temperature
    response = retry(
        lambda: client.models.generate_content(
            model=configuration["model_id"],
            contents=prompt,
            config=types.GenerateContentConfig(**options),
        )
    )
    text, reasoning = [], []
    for candidate in (response.candidates or [])[:1]:
        for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
            if part.text:
                (reasoning if getattr(part, "thought", False) else text).append(
                    part.text
                )
    if not text:
        try:
            text = [response.text or ""]
        except (AttributeError, IndexError, TypeError, ValueError):
            text = []
    usage = response.usage_metadata
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    thinking_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
    accounting = configuration.get(
        "output_token_accounting",
        "total" if configuration.get("profile") == "repeated" else "answer",
    )
    return {
        "text": "".join(text),
        "raw_text": "".join(text),
        "reasoning": "".join(reasoning),
        "output_tokens": output_tokens + thinking_tokens
        if accounting == "total"
        else output_tokens,
        "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
        "usage": {
            "candidates_token_count": output_tokens,
            "thoughts_token_count": thinking_tokens,
            "total_output_tokens": output_tokens + thinking_tokens,
        },
        "model_id": getattr(response, "model_version", None)
        or configuration["model_id"],
        "provider": "gemini",
        "finish_reason": str(response.candidates[0].finish_reason)
        if response.candidates
        else None,
    }
