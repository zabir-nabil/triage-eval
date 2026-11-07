from importlib import import_module


def complete(
    configuration, prompt, max_tokens, temperature=None, seed=None, phase="solver"
):
    name = configuration["provider"]
    modules = {
        "vllm": "vllm_local",
        "openai": "openai_api",
        "openrouter": "openrouter_api",
        "deepseek": "deepseek_api",
        "gemini": "gemini_api",
        "bedrock": "bedrock_api",
    }
    if name not in modules:
        raise ValueError(f"Unknown provider: {name}")
    module = import_module(f"triage_eval.providers.{modules[name]}")
    return module.complete(configuration, prompt, max_tokens, temperature, seed, phase)
