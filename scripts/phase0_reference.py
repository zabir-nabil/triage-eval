import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from triage_eval.datasets import load_dataset
from triage_eval.io import read_records
from triage_eval.planning import (
    configuration_names,
    load_configuration,
    load_settings,
    run_reference,
)


def _run_ids(args, settings):
    if args.run is not None:
        return [args.run]
    count = args.runs
    if count is None:
        count = settings.get("runs", 3) if args.profile == "repeated" else 1
    start = 1 if args.profile == "repeated" else 0
    return range(start, start + count)


def main():
    parser = argparse.ArgumentParser(
        description="Generate and grade reference completions"
    )
    parser.add_argument("--configuration", nargs="+", required=True)
    parser.add_argument("--config")
    parser.add_argument("--models")
    parser.add_argument("--profile", choices=("main", "repeated"), default="main")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--problems")
    parser.add_argument("--output", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", type=int)
    group.add_argument("--runs", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--lcb-tests")
    args = parser.parse_args()
    if args.run is not None and args.run < 0:
        parser.error("--run must be non-negative")
    if args.runs is not None and args.runs < 1:
        parser.error("--runs must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
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
    if args.lcb_tests:
        settings["lcb_tests"] = args.lcb_tests
    supplied_problems = None
    if args.problems:
        supplied_problems = read_records(args.problems)
        if args.datasets:
            selected = set(args.datasets)
            supplied_problems = [
                problem
                for problem in supplied_problems
                if problem.get("dataset") in selected
            ]
    for configuration_key in configurations:
        configuration = load_configuration(configuration_key, args.profile, args.models)
        run_settings = deepcopy(settings)
        if supplied_problems is None:
            datasets = args.datasets or list(
                run_settings.get(
                    "datasets", ("aime", "gpqa_diamond", "livecodebench", "hle")
                )
            )
            configured = run_settings.get("datasets", {})
            if not isinstance(configured, Mapping):
                configured = {name: {} for name in configured}
            for name, overrides in configuration.get("dataset_overrides", {}).items():
                configured[name] = {**configured.get(name, {}), **overrides}
            run_settings["datasets"] = configured
            problems = [
                problem
                for dataset in datasets
                for problem in load_dataset(dataset, run_settings)
            ]
        else:
            problems = supplied_problems
        if args.limit is not None:
            problems = problems[: args.limit]
        if not problems:
            parser.error(f"No problems were selected for {configuration_key}")
        for run in _run_ids(args, run_settings):
            run_reference(
                problems=problems,
                configuration=configuration,
                settings=run_settings,
                output=args.output,
                run=run,
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
