import gc
import json
from copy import deepcopy

from .common import split_reasoning


_engine = None
_engine_key = None


def complete(
    configuration, prompt, max_tokens, temperature=None, seed=None, phase="solver"
):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    global _engine, _engine_key
    model_id = configuration["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    template_options = {}
    if "Qwen3" in model_id:
        template_options["enable_thinking"] = configuration["mode"] == "extended"
    if "gpt-oss" in model_id:
        template_options["reasoning_effort"] = configuration.get(
            "reasoning_effort", "medium"
        )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **template_options,
    )
    input_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
    context_length = int(configuration.get(f"{phase}_context_length", 32768))
    if phase == "planner" and input_tokens + max_tokens > context_length:
        context_length = int(
            configuration.get("planner_long_context_length", context_length)
        )
    if input_tokens >= context_length:
        raise ValueError(
            f"Prompt has {input_tokens} tokens but context limit is {context_length}"
        )
    requested = min(int(max_tokens), context_length - input_tokens)
    options = {
        "model": model_id,
        "max_model_len": context_length,
        "dtype": "auto",
        "trust_remote_code": True,
        "gpu_memory_utilization": configuration.get("gpu_memory_utilization", 0.9),
        "tensor_parallel_size": configuration.get(
            "tensor_parallel_size", max(1, torch.cuda.device_count())
        ),
    }
    if context_length > 32768 and "Qwen" in model_id:
        original_context = 32768 if "Qwen2.5" in model_id else 40960
        options["hf_overrides"] = {
            "rope_scaling": {
                "rope_type": "yarn",
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": original_context,
            }
        }
    key = json.dumps(options, sort_keys=True)
    if key != _engine_key:
        if _engine is not None:
            previous = _engine
            _engine = None
            _engine_key = None
            del previous
            gc.collect()
            torch.cuda.empty_cache()
        _engine = LLM(**options)
        _engine_key = key
    if configuration.get("sampling") == "model_default":
        parameters = deepcopy(_engine.get_default_sampling_params())
        parameters.max_tokens = requested
    else:
        parameters = SamplingParams(
            temperature=0.0 if temperature is None else temperature,
            max_tokens=requested,
            seed=seed,
        )
    result = _engine.generate([rendered], parameters, use_tqdm=False)[0]
    output = result.outputs[0]
    reasoning, text = split_reasoning(output.text)
    return {
        "text": text,
        "raw_text": output.text,
        "reasoning": reasoning,
        "output_tokens": len(output.token_ids),
        "input_tokens": len(result.prompt_token_ids),
        "usage": {
            "requested_max_tokens": int(max_tokens),
            "max_tokens": requested,
            "context_length": context_length,
            "temperature": parameters.temperature,
            "seed": parameters.seed,
        },
        "model_id": model_id,
        "provider": "vllm",
        "finish_reason": output.finish_reason,
    }
