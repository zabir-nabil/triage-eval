from .common import compatible_completion


def complete(
    configuration, prompt, max_tokens, temperature=None, seed=None, phase="solver"
):
    return compatible_completion(
        configuration,
        prompt,
        max_tokens,
        temperature,
        "https://openrouter.ai/api/v1",
        seed,
        phase,
    )
