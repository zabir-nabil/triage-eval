import argparse
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from triage_eval.aggregation import aggregate_scores
from triage_eval.io import read_json, read_records, write_json
from triage_eval.planning import load_configuration, run_planning, select_problems
from triage_eval.scoring import score_pool


def control_configuration(configuration):
    name = configuration["name"].lower()
    if not any(
        family in name
        for family in ("gemini", "gpt-5 mini", "claude haiku", "kimi", "deepseek")
    ):
        raise ValueError(
            "The reference-solved control uses Claude, DeepSeek, Gemini, GPT-5 Mini, and Kimi configurations"
        )
    selected = deepcopy(configuration)
    if any(family in name for family in ("claude", "kimi", "deepseek")):
        selected = load_configuration(configuration["id"], "repeated")
        selected["profile"] = configuration.get("profile", "main")
    selected.update(
        temperature=None,
        planner_temperature=None,
        planner_max_tokens=24576,
        sampling="provider_default",
        output_token_accounting="total",
    )
    if "claude" in name and selected["mode"] == "extended":
        selected["thinking"] = {"type": "adaptive"}
    return selected


def run(
    problems,
    references,
    configuration,
    settings,
    alphas=(0.5, 0.75),
    run_id=0,
    seed=42,
    pool_size=30,
    min_pool=15,
):
    if pool_size <= 0 or not 1 <= min_pool <= pool_size:
        raise ValueError(
            "Pool sizes must be positive and minimum pool size cannot exceed pool size"
        )
    configuration = control_configuration(configuration)
    problems = select_problems(problems, configuration["id"])
    matched = [row for row in references if row["configuration"] == configuration["id"]]
    lookup = {(problem["dataset"], problem["id"]): problem for problem in problems}
    grouped = defaultdict(list)
    for row in matched:
        key = (row["dataset"], row["id"])
        if (
            row["dataset"] in ("aime", "gpqa_diamond", "livecodebench")
            and row.get("y") == 1
            and row.get("c", 0) > 0
            and key in lookup
        ):
            grouped[row["dataset"]].append(lookup[key])
    selected = []
    pools = []
    for dataset in ("aime", "gpqa_diamond", "livecodebench"):
        ordered = list(grouped[dataset])
        random.Random(seed).shuffle(ordered)
        for offset in range(0, len(ordered), pool_size):
            chunk = ordered[offset : offset + pool_size]
            if len(chunk) < min_pool:
                continue
            selected.extend(chunk)
            pools.append(
                {
                    "dataset": dataset,
                    "category": None,
                    "pool_id": f"{dataset}_{offset // pool_size:03d}",
                    "problem_ids": [problem["id"] for problem in chunk],
                }
            )
    if not selected:
        raise ValueError("No reference-solved pool meets the minimum pool size")
    scoped_settings = dict(settings, pools=pools)
    plans = run_planning(
        selected,
        matched,
        configuration,
        scoped_settings,
        run=run_id,
        alphas=list(alphas),
    )
    scores = [score_pool(plan, matched) for plan in plans]
    return {
        "n_problems": len(selected),
        "records": plans,
        "scores": scores,
        "summary": aggregate_scores(scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument(
        "--settings",
        default=str(Path(__file__).resolve().parents[1] / "configs/main.json"),
    )
    parser.add_argument("--profile", choices=("main", "repeated"), default="main")
    parser.add_argument("--reference-run", type=int, default=0)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--min-pool", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    references = [
        row
        for row in read_records(args.references)
        if row.get("run", 0) == args.reference_run
    ]
    result = run(
        read_records(args.problems),
        references,
        load_configuration(args.configuration, args.profile),
        read_json(args.settings),
        args.alphas,
        args.run,
        args.seed,
        args.pool_size,
        args.min_pool,
    )
    write_json(args.output, result)


if __name__ == "__main__":
    main()
