import argparse
import json
from pathlib import Path

from triage_eval.io import read_records
from triage_eval.planning import (
    configuration_names,
    load_configuration,
    load_settings,
    run_planning,
)


def _run_ids(args, settings):
    if args.run is not None:
        return [args.run]
    count = args.runs
    if count is None:
        count = settings.get("runs", 3) if args.profile == "repeated" else 1
    start = 1 if args.profile == "repeated" else 0
    return range(start, start + count)


def _problems_from_references(references):
    problems = {}
    for reference in references:
        if "text" not in reference:
            raise ValueError(
                "Reference records have no problem text; provide --problems"
            )
        key = (reference["dataset"], str(reference["id"]))
        problem = {
            field: reference[field]
            for field in ("id", "dataset", "category", "text", "metadata")
            if field in reference
        }
        if key in problems and problems[key].get("text") != problem.get("text"):
            raise ValueError(
                f"Conflicting problem text for {reference['id']}; provide --problems"
            )
        problems.setdefault(key, problem)
    return list(problems.values())


def main():
    parser = argparse.ArgumentParser(
        description="Generate task-pool plans from fixed references"
    )
    parser.add_argument("--references", required=True)
    parser.add_argument("--problems")
    parser.add_argument("--base-plans")
    parser.add_argument("--configuration", nargs="+", required=True)
    parser.add_argument("--config")
    parser.add_argument("--models")
    parser.add_argument("--profile", choices=("main", "repeated"), default="main")
    parser.add_argument("--reference-run", type=int, default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", type=int)
    group.add_argument("--runs", type=int)
    parser.add_argument("--alphas", nargs="+", type=float)
    parser.add_argument(
        "--variant", choices=("neutral", "agent", "exam"), default="neutral"
    )
    parser.add_argument("--planner-max-tokens", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.run is not None and args.run < 0:
        parser.error("--run must be non-negative")
    if args.runs is not None and args.runs < 1:
        parser.error("--runs must be positive")
    if args.reference_run < 0:
        parser.error("--reference-run must be non-negative")
    if args.alphas and any(not 0 < alpha <= 1 for alpha in args.alphas):
        parser.error("--alphas must be in (0, 1]")
    if args.planner_max_tokens is not None and args.planner_max_tokens < 1:
        parser.error("--planner-max-tokens must be positive")
    if "all" in args.configuration and args.configuration != ["all"]:
        parser.error("Use --configuration all without additional configuration names")
    configurations = (
        configuration_names(args.models)
        if args.configuration == ["all"]
        else list(dict.fromkeys(args.configuration))
    )
    settings = load_settings(args.config, profile=args.profile)
    if settings.get("profile", args.profile) != args.profile:
        parser.error("--profile must match the selected configuration file")
    settings["reference_run"] = args.reference_run
    if args.planner_max_tokens is not None:
        settings["planner_max_tokens"] = args.planner_max_tokens
    references = [
        row
        for row in read_records(args.references)
        if row.get("run", 0) == args.reference_run
    ]
    if not references:
        parser.error("No reference records match --reference-run")
    supplied_problems = read_records(args.problems) if args.problems else None
    base_plans = read_records(args.base_plans) if args.base_plans else None
    for configuration_key in configurations:
        configuration = load_configuration(configuration_key, args.profile, args.models)
        selected_references = [
            row for row in references if row.get("configuration") == configuration_key
        ]
        if not selected_references:
            parser.error(f"No references found for {configuration_key}")
        problems = (
            supplied_problems
            if supplied_problems is not None
            else _problems_from_references(selected_references)
        )
        run_settings = dict(settings)
        if base_plans is not None:
            frozen = [
                row
                for row in base_plans
                if row.get("configuration") == configuration_key
                and row.get("profile", "main") == "main"
                and row.get("run", 0) == 0
                and row.get("variant", "neutral") == args.variant
            ]
            if not frozen:
                parser.error(
                    f"No main base plans found for {configuration_key} and {args.variant}"
                )
            run_settings["pools"] = [
                {
                    field: row[field]
                    for field in (
                        "dataset",
                        "category",
                        "pool_id",
                        "alpha",
                        "budget",
                        "problem_ids",
                        "prompt",
                        "variant",
                        "usage",
                    )
                    if field in row
                }
                for row in frozen
            ]
        for run in _run_ids(args, settings):
            run_planning(
                problems=problems,
                references=selected_references,
                configuration=configuration,
                settings=run_settings,
                output=args.output,
                run=run,
                alphas=args.alphas,
                variant=args.variant,
            )
    print(
        json.dumps(
            {
                "output": str(Path(args.output)),
                "records": len(read_records(args.output)),
            }
        )
    )


if __name__ == "__main__":
    main()
