import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from triage_eval.grading import grade, load_livecodebench_tests
from triage_eval.io import read_json, read_records, write_json
from triage_eval.planning import (
    build_solver_prompt,
    load_configuration,
    request_completion,
)


def build_budget_aware_prompt(solver_prompt, allocation):
    return (
        f"You have a strict output-token budget of {allocation} tokens for solving "
        "this problem, including any reasoning, drafts, or scratch work you "
        "produce. Plan your solution to fit within this limit. Be concise, "
        "skip unnecessary explanation, and commit to your final answer before "
        "exhausting the budget.\n\n----\n\n" + solver_prompt
    )


def is_budget_aware_configuration(configuration):
    model = str(configuration["model_id"]).lower()
    return configuration.get("mode") == "standard" and any(
        name in model for name in ("gemini-2.5-flash", "kimi-k2.5", "glm-5.1")
    )


def index_problems(problems):
    result = {}
    for problem in problems:
        key = (problem.get("configuration"), problem["dataset"], str(problem["id"]))
        previous = result.get(key)
        if previous is not None and (
            previous["text"] != problem["text"]
            or previous.get("metadata", {}) != problem.get("metadata", {})
        ):
            raise ValueError(f"Conflicting problem records: {key}")
        result[key] = problem
    return result


def execution_settings(settings):
    names = (
        "solver_max_tokens",
        "planner_max_tokens",
        "seed",
        "seeds",
        "solver_context_length",
        "planner_context_length",
        "planner_long_context_length",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "lcb_tests",
    )
    return {name: settings[name] for name in names if name in settings}


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        if row.get("y") in (0, 1) and row.get("baseline_y") in (0, 1):
            groups[(row["configuration"], row["dataset"])].append(row)
    cells = []
    for (configuration, dataset), rows in sorted(groups.items()):
        counts = Counter()
        for row in rows:
            name = (
                "kept_correct"
                if row["baseline_y"] and row["y"]
                else "lost_correct"
                if row["baseline_y"]
                else "newly_correct"
                if row["y"]
                else "still_wrong"
            )
            counts[name] += 1
        cells.append(
            {
                "configuration": configuration,
                "dataset": dataset,
                "n": len(rows),
                "baseline_accuracy": mean(row["baseline_y"] for row in rows),
                "budget_aware_accuracy": mean(row["y"] for row in rows),
                "compliance": mean(row["c"] <= row["allocation"] for row in rows),
                **{
                    name: counts[name]
                    for name in (
                        "newly_correct",
                        "lost_correct",
                        "kept_correct",
                        "still_wrong",
                    )
                },
            }
        )
    models = []
    by_model = defaultdict(list)
    for key, rows in groups.items():
        by_model[key[0]].extend(rows)
    for configuration, rows in sorted(by_model.items()):
        counts = Counter()
        for row in rows:
            name = (
                "kept_correct"
                if row["baseline_y"] and row["y"]
                else "lost_correct"
                if row["baseline_y"]
                else "newly_correct"
                if row["y"]
                else "still_wrong"
            )
            counts[name] += 1
        models.append(
            {
                "configuration": configuration,
                "n": len(rows),
                "baseline_accuracy": mean(row["baseline_y"] for row in rows),
                "budget_aware_accuracy": mean(row["y"] for row in rows),
                "compliance": mean(row["c"] <= row["allocation"] for row in rows),
                **{
                    name: counts[name]
                    for name in (
                        "newly_correct",
                        "lost_correct",
                        "kept_correct",
                        "still_wrong",
                    )
                },
            }
        )
    return {
        "cells": cells,
        "models": models,
        "ungradeable": sum(
            row.get("y") is None or row.get("baseline_y") is None for row in records
        ),
    }


def run(
    plans,
    references,
    problems,
    settings,
    profile="main",
    alpha=1.0,
    output=None,
    test_cases=None,
    configurations=None,
    force=False,
    plan_run=0,
):
    lookup = index_problems(problems)
    baselines = {
        (row["configuration"], row["dataset"], row["id"]): row for row in references
    }
    records = (
        read_records(output) if output and Path(output).exists() and not force else []
    )
    if any(
        row.get("profile", "main") != profile or row.get("run", 0) != plan_run
        for row in records
    ):
        raise ValueError(
            "Use separate output files for different inference profiles and planning runs"
        )
    completed = {
        (
            row["configuration"],
            row["dataset"],
            row["pool_id"],
            row.get("run", 0),
            row["id"],
            row["allocation"],
        ): row
        for row in records
    }
    for plan in plans:
        if (
            plan.get("profile", "main") != profile
            or plan.get("variant", "neutral") != "neutral"
            or plan.get("run", 0) != plan_run
        ):
            continue
        if not math.isclose(float(plan["alpha"]), alpha) or not plan.get(
            "parsed_plan", {}
        ).get("ok"):
            continue
        if configurations and plan["configuration"] not in configurations:
            continue
        configuration = load_configuration(plan["configuration"], profile)
        if not is_budget_aware_configuration(configuration):
            continue
        for item in plan["parsed_plan"]["plan"]:
            allocation = item["tokens"]
            if allocation <= 0:
                continue
            key = (
                plan["configuration"],
                plan["dataset"],
                plan["pool_id"],
                plan.get("run", 0),
                item["id"],
                allocation,
            )
            problem = lookup.get(
                (plan["configuration"], plan["dataset"], item["id"]),
                lookup.get((None, plan["dataset"], item["id"])),
            )
            baseline = baselines.get(
                (plan["configuration"], plan["dataset"], item["id"])
            )
            if (
                problem is None
                or baseline is None
                or baseline.get("y") is None
                or not baseline.get("c", 0) > 0
            ):
                continue
            if "text" in baseline and (
                problem["text"] != baseline["text"]
                or problem.get("metadata", {}) != baseline.get("metadata", {})
            ):
                raise ValueError(
                    "Problem text and grading metadata must match the selected reference"
                )
            prompt = build_budget_aware_prompt(build_solver_prompt(problem), allocation)
            if key in completed:
                previous = completed[key]
                if (
                    previous.get("prompt") != prompt
                    or previous.get("configuration_settings") != configuration
                    or previous.get("baseline_y") != baseline["y"]
                    or previous.get("baseline_c") != baseline["c"]
                    or previous.get("execution_settings")
                    != execution_settings(settings)
                ):
                    raise ValueError(
                        "Existing re-solve does not match this prompt, configuration, and reference"
                    )
                continue
            response = request_completion(
                configuration,
                prompt,
                phase="solver",
                settings=settings,
                dataset=problem["dataset"],
                run=plan.get("run", 0),
            )
            row = {
                field: plan.get(field)
                for field in (
                    "configuration",
                    "dataset",
                    "category",
                    "pool_id",
                    "alpha",
                    "run",
                )
            }
            row["profile"] = profile
            row.update(
                {
                    "id": item["id"],
                    "allocation": allocation,
                    "prompt": prompt,
                    "configuration_settings": configuration,
                    "execution_settings": execution_settings(settings),
                    "baseline_y": baseline["y"],
                    "baseline_c": baseline["c"],
                    "y": grade(problem, response["text"], test_cases),
                    "c": response["output_tokens"],
                    "response": response["text"],
                    "reasoning": response.get("reasoning", ""),
                    "usage": {
                        name: value
                        for name, value in response.items()
                        if name not in ("text", "reasoning", "raw_text")
                    },
                }
            )
            records.append(row)
            completed[key] = row
            if output:
                write_json(output, {"records": records, "summary": summarize(records)})
    result = {"records": records, "summary": summarize(records)}
    if output:
        write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results")
    source.add_argument("--plans")
    parser.add_argument("--references")
    parser.add_argument("--problems")
    parser.add_argument("--configuration", nargs="+")
    parser.add_argument(
        "--settings",
        default=str(Path(__file__).resolve().parents[1] / "configs/main.json"),
    )
    parser.add_argument("--profile", choices=("main", "repeated"), default="main")
    parser.add_argument("--reference-run", type=int, default=0)
    parser.add_argument("--plan-run", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lcb-tests")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.results:
        records = read_records(args.results)
        write_json(args.output, {"records": records, "summary": summarize(records)})
        return
    if not args.references or not args.problems:
        parser.error("--references and --problems are required with --plans")
    references = [
        row
        for row in read_records(args.references)
        if row.get("run", 0) == args.reference_run
    ]
    plans = read_records(args.plans)
    tests = (
        load_livecodebench_tests(args.lcb_tests)
        if args.lcb_tests
        or any(plan.get("dataset") == "livecodebench" for plan in plans)
        else None
    )
    run(
        plans,
        references,
        read_records(args.problems),
        read_json(args.settings),
        args.profile,
        args.alpha,
        args.output,
        tests,
        args.configuration,
        args.force,
        args.plan_run,
    )


if __name__ == "__main__":
    main()
