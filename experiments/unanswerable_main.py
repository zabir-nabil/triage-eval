import argparse
import math
import random
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from statistics import mean

from triage_eval.datasets import load_unanswerable
from triage_eval.io import read_json, read_records, write_json
from triage_eval.parsing import parse_plan
from triage_eval.planning import (
    build_prompt,
    load_configuration,
    request_completion,
    run_reference,
    sampling_settings,
    select_problems,
)


def probe_configuration(configuration, listed=False):
    pilot = {
        "gemini-2.5-flash-extended",
        "qwen2.5-32b-standard",
        "qwen3-32b-extended",
        "gpt-oss-20b-extended",
    }
    retained = pilot - {"qwen3-32b-extended"}
    identifier = configuration["id"]
    if not listed and identifier not in pilot:
        raise ValueError(
            "The main probe uses Gemini extended, Qwen 2.5 32B, Qwen 3 32B, and GPT-OSS 20B"
        )
    selected = deepcopy(configuration)
    if not listed or identifier in retained:
        selected = load_configuration(identifier, "main")
        selected["profile"] = configuration.get("profile", "main")
        if identifier.startswith("gemini"):
            selected.update(
                temperature=None,
                thinking_budget=24576,
                solver_max_tokens=24576,
                planner_max_tokens=24576,
                output_token_accounting="answer",
            )
        elif identifier.startswith("gpt-oss"):
            selected.update(
                temperature=0.0,
                solver_max_tokens=32768,
                planner_max_tokens=32768,
                solver_context_length=131072,
                planner_context_length=131072,
                planner_long_context_length=131072,
            )
        else:
            limit = 4096 if identifier.startswith("qwen2.5") else 24576
            selected.update(
                temperature=0.0,
                solver_max_tokens=limit,
                planner_max_tokens=limit,
                solver_context_length=32768,
                planner_context_length=32768,
                planner_long_context_length=32768,
            )
        return selected
    name = selected["name"].lower()
    if any(
        family in name for family in ("claude", "kimi", "deepseek")
    ) or identifier.startswith("gpt-oss-120b"):
        selected = load_configuration(identifier, "repeated")
        selected["profile"] = configuration.get("profile", "main")
    selected.update(
        temperature=None,
        solver_temperature=None,
        planner_temperature=None,
        output_token_accounting="total",
    )
    if selected["provider"] == "vllm":
        limit = 4096 if selected["mode"] == "standard" else 24576
        selected.update(
            sampling="model_default",
            solver_max_tokens=limit,
            planner_max_tokens=limit,
            solver_context_length=32768,
            planner_context_length=32768,
            planner_long_context_length=32768,
        )
    else:
        selected.update(
            sampling="provider_default",
            solver_max_tokens=24576,
            planner_max_tokens=24576,
        )
        if "claude" in name and selected["mode"] == "extended":
            selected["thinking"] = {"type": "adaptive"}
        if "glm" in name and selected["mode"] == "standard":
            selected["extra_body"] = {"reasoning": {"enabled": False}}
    return selected


def is_unanswerable(problem):
    return bool(
        problem.get(
            "unanswerable", problem.get("metadata", {}).get("unanswerable", False)
        )
    )


def build_mix(problems, ratio, seed=42):
    if not 0 <= ratio <= 1:
        raise ValueError("Injection ratio must lie between zero and one")
    answerable = [problem for problem in problems if not is_unanswerable(problem)]
    unanswerable = [problem for problem in problems if is_unanswerable(problem)]
    count = len(answerable)
    injected = int(round(ratio * count))
    if not count or injected > len(unanswerable):
        raise ValueError(
            "The source does not contain enough paired answerable and unanswerable items"
        )
    selection = random.Random(seed * 100003 + int(ratio * 1000))
    mixed = selection.sample(answerable, count - injected) + selection.sample(
        unanswerable, injected
    )
    random.Random(seed * 100019 + int(ratio * 1000)).shuffle(mixed)
    return mixed


def detection_metrics(plan, problem_ids, unanswerable_ids, listed=False):
    pool = set(problem_ids)
    injected = set(unanswerable_ids) & pool
    allocations = {}
    for entry in plan:
        identifier = str(entry["id"])
        if identifier in pool and identifier not in allocations:
            allocations[identifier] = max(0, int(entry.get("tokens", 0)))
    selected = (
        set(allocations)
        if listed
        else {pid for pid, tokens in allocations.items() if tokens > 0}
    )
    total = sum(allocations.values())
    detection = len(injected - selected) / len(injected) if injected else None
    waste = (
        sum(tokens for pid, tokens in allocations.items() if pid in injected) / total
        if total
        else 0.0
    )
    answerable = pool - injected
    coverage = (
        len(set(allocations) & answerable) / len(answerable)
        if listed and answerable
        else None
    )
    discrimination = (
        detection + coverage - 1
        if detection is not None and coverage is not None
        else None
    )
    return {
        "detection": detection,
        "waste": waste,
        "coverage": coverage,
        "discrimination": discrimination,
    }


def analyze(plans, listed=False):
    cohorts = {(row.get("profile", "main"), row.get("run", 0)) for row in plans}
    if len(cohorts) > 1:
        raise ValueError("Select one inference profile and planning run")
    grouped = defaultdict(list)
    pools = []
    identities = {}
    for record in plans:
        parsed = record.get("parsed_plan", {})
        if not parsed.get("ok"):
            continue
        if not listed and record.get("budget", 0) <= 0:
            continue
        values = detection_metrics(
            parsed["plan"], record["problem_ids"], record["unanswerable_ids"], listed
        )
        if values["detection"] is None:
            continue
        row = {
            key: record.get(key)
            for key in ("configuration", "dataset", "pool_id", "ratio", "run")
        }
        row.update(values)
        pools.append(row)
        configuration = record.get("configuration_settings", {})
        identities[record["configuration"]] = {
            "model": configuration.get(
                "name", record["configuration"].rsplit("-", 1)[0]
            ),
            "mode": record.get("mode", configuration.get("mode")),
        }
        grouped[(record["configuration"], record["dataset"], record["ratio"])].append(
            values
        )
    metrics = ("detection", "waste", "coverage", "discrimination")
    ratios = []
    for (configuration, dataset, ratio), rows in sorted(grouped.items()):
        output = {
            "configuration": configuration,
            "dataset": dataset,
            "ratio": ratio,
            "n_pools": len(rows),
        }
        for metric in metrics:
            values = [row[metric] for row in rows if row[metric] is not None]
            output[metric] = mean(values) if values else None
        ratios.append(output)
    grouped_ratios = defaultdict(list)
    for row in ratios:
        grouped_ratios[(row["configuration"], row["dataset"])].append(row)
    configurations = []
    for (configuration, dataset), rows in sorted(grouped_ratios.items()):
        output = {
            "configuration": configuration,
            "dataset": dataset,
            **identities.get(configuration, {}),
        }
        for metric in metrics:
            values = [row[metric] for row in rows if row[metric] is not None]
            output[metric] = mean(values) if values else None
        configurations.append(output)
    overall = {
        metric: mean(row[metric] for row in configurations if row[metric] is not None)
        if any(row[metric] is not None for row in configurations)
        else None
        for metric in metrics
    }
    pairs = defaultdict(dict)
    for row in configurations:
        if row.get("mode") in ("standard", "extended"):
            pairs[(row["model"], row["dataset"])][row["mode"]] = row
    matched = []
    for (model, dataset), pair in sorted(pairs.items()):
        if set(pair) == {"standard", "extended"}:
            matched.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "standard": {
                        metric: pair["standard"][metric] for metric in metrics
                    },
                    "extended": {
                        metric: pair["extended"][metric] for metric in metrics
                    },
                }
            )
    matched_means = {}
    for mode in ("standard", "extended"):
        matched_means[mode] = {
            metric: mean(
                row[mode][metric] for row in matched if row[mode][metric] is not None
            )
            if any(row[mode][metric] is not None for row in matched)
            else None
            for metric in metrics
        }
    return {
        "selection": "listed" if listed else "positive_allocation",
        "pools": pools,
        "ratios": ratios,
        "configurations": configurations,
        "overall": overall,
        "matched": matched,
        "matched_means": matched_means,
    }


def run_probe(
    problems,
    references,
    configuration,
    settings,
    ratios=(0.25, 0.5, 0.75, 1.0),
    seed=42,
    pool_size=10,
    run_id=0,
    listed=False,
    output=None,
):
    if pool_size <= 0:
        raise ValueError("Pool size must be positive")
    configuration = probe_configuration(configuration, listed)
    problems = select_problems(problems, configuration["id"])
    existing = read_json(output) if output and Path(output).exists() else {}
    selection = "listed" if listed else "positive_allocation"
    if existing.get("selection", selection) != selection:
        raise ValueError("Main and expanded probe outputs must use separate files")
    plans = existing.get("records", [])
    profile = configuration.get("profile", "main")
    if any(
        row.get("profile", "main") != profile or row.get("run", 0) != run_id
        for row in plans
    ):
        raise ValueError(
            "Use separate output files for different inference profiles and planning runs"
        )
    request_settings = sampling_settings(configuration, settings, "planner", run_id)
    stored_references = existing.get("references", [])
    if references is None:
        references = [
            row
            for row in stored_references
            if row["configuration"] == configuration["id"]
        ]
        if not references:
            references = run_reference(problems, configuration, settings, run=0)
    references = [
        row for row in references if row["configuration"] == configuration["id"]
    ]
    reference_keys = {
        (row["configuration"], row["dataset"], row["id"], row.get("run", 0))
        for row in stored_references
    }
    stored_references.extend(
        row
        for row in references
        if (row["configuration"], row["dataset"], row["id"], row.get("run", 0))
        not in reference_keys
    )
    completed = {
        (
            row["configuration"],
            row["dataset"],
            row["ratio"],
            row["pool_id"],
            row.get("run", 0),
        ): row
        for row in plans
    }
    measured = {
        (row["dataset"], row["id"]): row for row in references if row.get("c", 0) > 0
    }
    datasets = defaultdict(list)
    for problem in problems:
        datasets[problem["dataset"]].append(problem)
    for dataset, source in sorted(datasets.items()):
        for ratio in ratios:
            mixed = build_mix(source, ratio, seed)
            for index in range(0, len(mixed), pool_size):
                chunk = mixed[index : index + pool_size]
                missing = [
                    problem["id"]
                    for problem in chunk
                    if (dataset, problem["id"]) not in measured
                ]
                if missing:
                    raise ValueError(
                        f"Missing positive reference costs for {len(missing)} injected-pool items"
                    )
                budget = math.floor(
                    sum(measured[(dataset, problem["id"])]["c"] for problem in chunk)
                )
                identifiers = [problem["id"] for problem in chunk]
                prompt = build_prompt(
                    chunk,
                    budget,
                    domain="graduate-level science questions"
                    if dataset.startswith("gpqa")
                    else "math questions",
                )
                pool_id = f"{dataset}_{index // pool_size:03d}"
                identity = (configuration["id"], dataset, ratio, pool_id, run_id)
                if identity in completed:
                    previous = completed[identity]
                    if (
                        previous.get("prompt") != prompt
                        or previous.get("configuration_settings") != configuration
                        or previous.get("sampling_settings") != request_settings
                    ):
                        raise ValueError(
                            "Existing probe plan does not match this prompt and configuration"
                        )
                    continue
                response = request_completion(
                    configuration,
                    prompt,
                    phase="planner",
                    settings=settings,
                    dataset=dataset,
                    run=run_id,
                )
                record = {
                    "configuration": configuration["id"],
                    "mode": configuration.get("mode"),
                    "configuration_settings": configuration,
                    "dataset": dataset,
                    "category": None,
                    "pool_id": pool_id,
                    "alpha": 1.0,
                    "budget": budget,
                    "run": run_id,
                    "ratio": ratio,
                    "problem_ids": identifiers,
                    "unanswerable_ids": [
                        problem["id"] for problem in chunk if is_unanswerable(problem)
                    ],
                    "prompt": prompt,
                    "response": response["text"],
                    "reasoning": response.get("reasoning", ""),
                    "usage": {
                        key: value
                        for key, value in response.items()
                        if key not in ("text", "reasoning", "raw_text")
                    },
                    "parsed_plan": parse_plan(response["text"], identifiers),
                }
                record.update(profile=profile, sampling_settings=request_settings)
                plans.append(record)
                completed[identity] = record
                if output:
                    write_json(
                        output,
                        {
                            "selection": selection,
                            "records": plans,
                            "references": stored_references,
                            "summary": analyze(plans, listed),
                        },
                    )
    result = {
        "selection": selection,
        "records": plans,
        "references": stored_references,
        "summary": analyze(plans, listed),
    }
    if output:
        write_json(output, result)
    return result


def command(listed=False):
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--plans")
    source.add_argument("--problems")
    parser.add_argument("--references")
    parser.add_argument("--reference-run", type=int, default=0)
    parser.add_argument("--configuration")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("gpqa", "mmlu_math"),
        default=["gpqa"] if listed else ["gpqa", "mmlu_math"],
    )
    parser.add_argument(
        "--settings",
        default=str(Path(__file__).resolve().parents[1] / "configs/main.json"),
    )
    parser.add_argument("--profile", choices=("main", "repeated"), default="main")
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.plans:
        write_json(args.output, analyze(read_records(args.plans), listed))
        return
    if not args.configuration:
        parser.error("--configuration is required when generating plans")
    problems = (
        read_records(args.problems)
        if args.problems
        else [problem for name in args.datasets for problem in load_unanswerable(name)]
    )
    references = (
        [
            row
            for row in read_records(args.references)
            if row.get("run", 0) == args.reference_run
        ]
        if args.references
        else None
    )
    run_probe(
        problems,
        references,
        load_configuration(args.configuration, args.profile),
        read_json(args.settings),
        args.ratios,
        args.seed,
        args.pool_size,
        args.run,
        listed,
        args.output,
    )


if __name__ == "__main__":
    command()
