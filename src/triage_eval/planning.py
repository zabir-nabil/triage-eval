from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

from .grading import grade, load_livecodebench_tests
from .io import read_json, read_records, write_json, write_jsonl
from .parsing import parse_plan
from .pools import build_pools, calculate_budget
from .providers import complete


def resource_path(name):
    candidates = [
        Path(__file__).resolve().parents[2] / name,
        Path(sys.prefix) / "share" / "triage-eval" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


def load_settings(path=None, profile="main"):
    result = read_json(path or resource_path(f"configs/{profile}.json"))
    if not isinstance(result, dict):
        raise ValueError("Settings must be a JSON object")
    return result


def load_configuration(key, profile="main", path=None):
    if profile not in {"main", "repeated"}:
        raise ValueError(f"Unknown profile: {profile}")
    configurations = read_json(path or resource_path("configs/models.json"))[
        "configurations"
    ]
    if key not in configurations:
        raise ValueError(
            f"Unknown configuration {key}; choose from {', '.join(configurations)}"
        )
    entry = deepcopy(configurations[key])
    result = {
        name: value for name, value in entry.items() if name not in {"main", "repeated"}
    }
    result.update(entry[profile])
    result.update(id=key, profile=profile)
    return result


def configuration_names(path=None):
    return list(
        read_json(path or resource_path("configs/models.json"))["configurations"]
    )


def select_problems(problems, configuration=None):
    result = {}
    for problem in problems:
        if (
            configuration is not None
            and problem.get("configuration", configuration) != configuration
        ):
            continue
        row = {name: problem[name] for name in ("id", "dataset", "text")}
        row["id"] = str(row["id"])
        row["category"] = problem.get("category")
        row["metadata"] = problem.get("metadata", {})
        key = (row["dataset"], row["id"])
        if key in result and result[key] != row:
            raise ValueError(f"Conflicting problem records: {key}")
        result[key] = row
    return list(result.values())


def build_solver_prompt(problem):
    dataset = problem["dataset"]
    text = problem["text"]
    metadata = problem.get("metadata", {})
    if "abstain" in dataset or "unanswerable" in dataset:
        return (
            'Answer the following question. Reason step by step and end your response with: "The answer is (insert answer here)."\n\n'
            + text
        )
    if dataset == "aime":
        return (
            "Solve the following math problem step by step. The answer is an integer between 000 and 999 inclusive.\n\n"
            + text
            + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
    if dataset == "gpqa_diamond":
        if metadata.get("choices"):
            choices = "\n".join(
                f"({chr(65 + index)}) {choice}"
                for index, choice in enumerate(metadata["choices"])
            )
            text = f"{metadata['question']}\n\nChoices:\n{choices}"
        return f'What is the correct answer to this question: {text}\n\nFormat your response as follows: "The correct answer is (insert answer here)".'
    if dataset == "livecodebench":
        text = metadata.get("question_content", text)
        prompt = (
            "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. Put your final solution within code delimiters, for example:\n```python\n{your code}\n```\n\n### Question:\n"
            + text
        )
        if metadata.get("starter_code"):
            prompt += "\n\n### Starter Code:\n" + metadata["starter_code"]
        return prompt
    if dataset == "hle":
        if metadata.get("answer_type") == "multiple_choice":
            return f'What is the correct answer to this question: {text}\n\nFormat your response as follows: "The correct answer is (insert answer here)".'
        return (
            'Answer the following question. Your response should end with "The answer is (insert answer here)".\n\n'
            + text
        )
    raise ValueError(f"No solver prompt for dataset {dataset}")


def build_prompt(problems, budget, domain=None, variant="neutral"):
    if not problems:
        raise ValueError("A planning pool must contain problems")
    if variant not in {"neutral", "agent", "exam"}:
        raise ValueError(f"Unknown prompt variant: {variant}")
    template = resource_path("prompts/planner.txt").read_text(encoding="utf-8-sig")
    if variant != "neutral":
        opening = (
            resource_path(f"prompts/variants/{variant}.txt")
            .read_text(encoding="utf-8-sig")
            .strip()
        )
        template = opening + "\n\n" + template.split("\n\n", 1)[1]
    domains = {
        "aime": "competition mathematics problems",
        "gpqa_diamond": "graduate-level multiple-choice science questions",
        "livecodebench": "competitive programming problems",
        "hle": "expert-level questions across multiple domains",
        "gpqa_abstain": "graduate-level science questions",
        "mmlu_math_abstain": "mathematics questions",
    }
    domain = domain or domains.get(problems[0]["dataset"], "problems")
    block = "\n\n".join(
        f"[{problem['id']}] (points: 1)\n{problem['text']}" for problem in problems
    )
    return (
        template.replace("{domain}", domain)
        .replace("{B}", str(int(budget)))
        .replace("{problem_block}", block)
    )


def request_completion(
    configuration, prompt, phase="solver", settings=None, dataset=None, run=0
):
    settings = settings or {}
    if phase not in {"solver", "planner"}:
        raise ValueError(f"Unknown phase: {phase}")
    if isinstance(configuration, str):
        configuration = load_configuration(
            configuration, settings.get("profile", "main")
        )
    selected = deepcopy(configuration)
    limit = settings.get(f"{phase}_max_tokens", selected[f"{phase}_max_tokens"])
    if isinstance(limit, dict):
        limit = limit.get(dataset, limit["default"])
    if isinstance(limit, list):
        raise ValueError("Choose one generation limit for this request")
    if int(limit) <= 0:
        raise ValueError("Generation limits must be positive")
    temperature = selected.get(
        f"{phase}_temperature",
        selected.get("temperature", settings.get("temperature", 0.0)),
    )
    seed = _seed(selected, settings, phase, run)
    for name in [
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "solver_context_length",
        "planner_context_length",
        "planner_long_context_length",
    ]:
        if name in settings:
            selected[name] = settings[name]
    result = complete(
        selected, prompt, int(limit), temperature=temperature, seed=seed, phase=phase
    )
    if (
        int(result.get("output_tokens", 0)) < 0
        or int(result.get("input_tokens", 0)) < 0
    ):
        raise ValueError("Provider returned a negative token count")
    result["sampling"] = {
        "max_tokens": int(limit),
        "temperature": temperature,
        "seed": seed,
    }
    return result


def _save_records(path, records):
    if path:
        if Path(path).suffix.lower() == ".jsonl":
            write_jsonl(path, records)
        else:
            write_json(path, {"records": records})


def _existing_records(output):
    return read_records(output) if output and Path(output).is_file() else []


def _seed(configuration, settings, phase, run):
    if configuration.get("provider") != "vllm":
        return None
    if configuration.get("sampling") == "model_default":
        return None
    if "seed" in configuration:
        return configuration["seed"]
    if "seed" in settings:
        return int(settings["seed"])
    seeds = settings.get("seeds", [1, 2, 3])
    if run > 0:
        return int(seeds[run - 1]) if run <= len(seeds) else run
    return (
        42
        if phase == "planner" and configuration.get("profile", "main") == "main"
        else None
    )


def sampling_settings(configuration, settings, phase="planner", run=0):
    names = [
        f"{phase}_max_tokens",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        f"{phase}_context_length",
    ]
    if phase == "planner":
        names.append("planner_long_context_length")
    result = {name: settings[name] for name in names if name in settings}
    result["seed"] = _seed(configuration, settings, phase, run)
    return result


def run_reference(problems, configuration, settings, output=None, run=0):
    if isinstance(configuration, str):
        configuration = load_configuration(
            configuration, settings.get("profile", "main")
        )
    configuration_id = configuration["id"]
    problems = select_problems(problems, configuration_id)
    profile = configuration.get("profile", settings.get("profile", "main"))
    run_settings = sampling_settings(configuration, settings, "solver", run)
    all_records = _existing_records(output)
    rows = [
        row
        for row in all_records
        if row.get("configuration") == configuration_id
        and row.get("run", 0) == run
        and row.get("profile", "main") == profile
    ]
    existing = {(row["dataset"], str(row["id"])): row for row in rows}
    database = None
    if any(problem["dataset"] == "livecodebench" for problem in problems):
        database = load_livecodebench_tests(settings.get("lcb_tests"))
    try:
        for index, problem in enumerate(problems):
            identity = (problem["dataset"], str(problem["id"]))
            if identity in existing:
                previous = existing[identity]
                if (
                    previous.get("text") != problem["text"]
                    or previous.get("metadata") != problem.get("metadata", {})
                    or previous.get("configuration_settings") != configuration
                    or previous.get("run_settings") != run_settings
                ):
                    raise ValueError(
                        "Existing reference record does not match this configuration and problem"
                    )
                continue
            prompt = build_solver_prompt(problem)
            completion = request_completion(
                configuration, prompt, "solver", settings, problem["dataset"], run
            )
            correct = (
                0
                if problem.get("metadata", {}).get("unanswerable")
                else grade(problem, completion["text"], database)
            )
            row = dict(problem)
            row.update(
                configuration=configuration_id,
                profile=profile,
                run=run,
                prompt=prompt,
                response=completion["text"],
                reasoning=completion.get("reasoning", ""),
                y=None if correct is None else int(correct),
                c=int(completion["output_tokens"]),
                usage=completion,
                configuration_settings=configuration,
                run_settings=run_settings,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            all_records.append(row)
            rows.append(row)
            existing[identity] = row
            if (index + 1) % settings.get("checkpoint_every", 20) == 0:
                _save_records(output, all_records)
    except BaseException:
        _save_records(output, all_records)
        raise
    _save_records(output, all_records)
    return rows


def run_planning(
    problems,
    references,
    configuration,
    settings,
    output=None,
    run=0,
    alphas=None,
    variant="neutral",
):
    if isinstance(configuration, str):
        configuration = load_configuration(
            configuration, settings.get("profile", "main")
        )
    configuration_id = configuration["id"]
    problems = select_problems(problems, configuration_id)
    profile = configuration.get("profile", settings.get("profile", "main"))
    reference_run = int(settings.get("reference_run", 0))
    scoped_references = [
        row
        for row in references
        if row.get("configuration", configuration_id) == configuration_id
        and row.get("run", 0) == reference_run
    ]
    reference_index = {}
    for row in scoped_references:
        identity = (row["dataset"], str(row["id"]))
        if identity in reference_index:
            raise ValueError(f"Duplicate reference record: {identity}")
        reference_index[identity] = row
    problem_index = {(row["dataset"], str(row["id"])): row for row in problems}
    pools = settings.get("pools") or build_pools(
        problems, settings.get("pool_size", 30), settings.get("min_tail", 10)
    )
    budgets = alphas if alphas is not None else settings.get("alphas", [0.5])
    all_records = _existing_records(output)
    rows = [
        row
        for row in all_records
        if row.get("configuration") == configuration_id
        and row.get("profile", "main") == profile
        and row.get("run", 0) == run
        and row.get("variant", "neutral") == variant
    ]
    existing = {
        (row["dataset"], str(row["pool_id"]), float(row["alpha"])): row for row in rows
    }
    for pool in pools:
        dataset = pool["dataset"]
        pool_problems = [
            problem_index[(dataset, str(pid))] for pid in pool["problem_ids"]
        ]
        costs = [
            reference_index[(dataset, str(problem["id"]))].get("c")
            for problem in pool_problems
            if (dataset, str(problem["id"])) in reference_index
        ]
        if not costs or not any(cost is not None and cost > 0 for cost in costs):
            continue
        for alpha in budgets:
            if "alpha" in pool and float(pool["alpha"]) != float(alpha):
                continue
            budget = (
                int(pool["budget"])
                if "budget" in pool
                else calculate_budget(costs, alpha)
            )
            prompt = pool.get("prompt") or build_prompt(
                pool_problems, budget, pool.get("domain"), variant
            )
            request_settings = dict(settings)
            previous_limit = pool.get("usage", {}).get("sampling", {}).get("max_tokens")
            if (
                profile == "repeated"
                and configuration["provider"] != "vllm"
                and previous_limit
            ):
                request_settings.setdefault("planner_max_tokens", previous_limit)
            run_settings = sampling_settings(
                configuration, request_settings, "planner", run
            )
            identity = (dataset, str(pool["pool_id"]), float(alpha))
            if identity in existing:
                previous = existing[identity]
                if (
                    previous.get("prompt") != prompt
                    or previous.get("configuration_settings") != configuration
                    or previous.get("run_settings") != run_settings
                ):
                    raise ValueError(
                        "Existing plan does not match this prompt and configuration"
                    )
                continue
            completion = request_completion(
                configuration, prompt, "planner", request_settings, dataset, run
            )
            valid_ids = [str(problem["id"]) for problem in pool_problems]
            parsed = parse_plan(completion["text"], valid_ids, recovery=True)
            if not parsed["ok"] and completion.get("raw_text") != completion["text"]:
                parsed = parse_plan(
                    completion.get("raw_text", ""), valid_ids, recovery=True
                )
            row = {
                "configuration": configuration_id,
                "profile": profile,
                "variant": variant,
                "dataset": dataset,
                "category": pool.get("category"),
                "pool_id": pool["pool_id"],
                "alpha": float(alpha),
                "budget": budget,
                "run": run,
                "reference_run": reference_run,
                "problem_ids": valid_ids,
                "prompt": prompt,
                "response": completion["text"],
                "parsed_plan": parsed,
                "usage": completion,
                "configuration_settings": configuration,
                "run_settings": run_settings,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            all_records.append(row)
            rows.append(row)
            existing[identity] = row
            _save_records(output, all_records)
    return rows
