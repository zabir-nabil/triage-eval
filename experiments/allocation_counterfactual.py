import argparse
import math
from collections import defaultdict
from statistics import mean, median

from triage_eval.io import read_records, write_json
from triage_eval.scoring import (
    compute_eta,
    compute_oracle,
    compute_random,
    compute_regret,
    execute_plan_E,
    execute_plan_U,
    get_budget,
    matched_problems,
)


def compare(
    plans, references, alpha=0.5, margin=0.2, random_samples=1000, random_seed=42
):
    if margin < 0:
        raise ValueError("Margin must be nonnegative")
    plans = [record for record in plans if math.isclose(float(record["alpha"]), alpha)]
    cohorts = {
        (
            record.get("profile", "main"),
            record.get("variant", "neutral"),
            record.get("run", 0),
        )
        for record in plans
    }
    if len(cohorts) > 1:
        raise ValueError(
            "Select one inference profile, prompt variant, and planning run"
        )
    rows = []
    excluded = {
        "unparseable": 0,
        "reference_coverage": 0,
        "empty_selection": 0,
        "no_usable_selection": 0,
    }
    for record in plans:
        if not math.isclose(float(record["alpha"]), alpha):
            continue
        parsed = record.get("parsed_plan", {})
        if not parsed.get("ok"):
            excluded["unparseable"] += 1
            continue
        problems = matched_problems(record, references)
        if not problems:
            excluded["reference_coverage"] += 1
            continue
        if not parsed.get("plan"):
            excluded["empty_selection"] += 1
            continue
        selected = [item for item in parsed["plan"] if item["id"] in problems]
        if not selected:
            excluded["no_usable_selection"] += 1
            continue
        budget = get_budget(record)
        order = [item["id"] for item in selected]
        oracle = compute_oracle(problems, budget)
        random = compute_random(
            problems, budget, n_samples=random_samples, rng_seed=random_seed
        )
        variants = {
            "declared": selected,
            "reference_cost": [
                {"id": pid, "tokens": problems[pid]["c"]} for pid in order
            ],
            "cost_margin": [
                {"id": pid, "tokens": math.ceil((1 + margin) * problems[pid]["c"])}
                for pid in order
            ],
            "equal_split": [
                {"id": pid, "tokens": max(1, budget // len(order))} for pid in order
            ],
        }
        values = {
            key: execute_plan_E(plan, problems, budget)
            for key, plan in variants.items()
        }
        values["advisory"] = execute_plan_U(order, problems, budget)
        row = {
            key: record.get(key)
            for key in (
                "configuration",
                "dataset",
                "category",
                "pool_id",
                "alpha",
                "run",
            )
        }
        row.update(
            {
                "budget": budget,
                "n_selected": len(order),
                "V_oracle": oracle,
                "V_random": random,
            }
        )
        for name, value in values.items():
            row[f"value_{name}"] = value
            row[f"eta_{name}"] = (
                compute_eta(value, random, oracle) if alpha < 1 else None
            )
            row[f"regret_{name}"] = compute_regret(value, oracle)
        ratios = [item["tokens"] / problems[item["id"]]["c"] for item in selected]
        row["underallocation_fraction"] = mean(value < 1 for value in ratios)
        row["allocation_cost_ratio_median"] = median(ratios)
        rows.append(row)
    grouped = defaultdict(lambda: defaultdict(list))
    metric_names = [
        f"{metric}_{name}"
        for metric in ("eta", "regret")
        for name in (
            "declared",
            "reference_cost",
            "cost_margin",
            "equal_split",
            "advisory",
        )
    ]
    for row in rows:
        for metric in metric_names:
            if row[metric] is not None:
                grouped[row["configuration"]][metric].append(row[metric])
    configurations = []
    for configuration, metrics in sorted(grouped.items()):
        output = {"configuration": configuration}
        for metric in metric_names:
            output[metric] = mean(metrics[metric]) if metrics[metric] else None
        configurations.append(output)
    overall = {
        metric: mean(row[metric] for row in configurations if row[metric] is not None)
        if any(row[metric] is not None for row in configurations)
        else None
        for metric in metric_names
    }
    gap = (
        None
        if overall["eta_advisory"] is None
        else overall["eta_advisory"] - overall["eta_declared"]
    )
    overall["gap_recovery"] = {
        name: (overall[f"eta_{name}"] - overall["eta_declared"]) / gap
        if gap and gap > 0
        else None
        for name in ("reference_cost", "cost_margin", "equal_split")
    }
    return {
        "alpha": alpha,
        "margin": margin,
        "pools": rows,
        "configurations": configurations,
        "overall": overall,
        "excluded": excluded,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--reference-run", type=int, default=0)
    parser.add_argument("--plan-run", type=int, default=0)
    parser.add_argument("--profile", choices=("main", "repeated"), default="main")
    parser.add_argument(
        "--variant", choices=("neutral", "agent", "exam"), default="neutral"
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    references = [
        row
        for row in read_records(args.references)
        if row.get("run", 0) == args.reference_run
    ]
    plans = [
        row
        for row in read_records(args.plans)
        if row.get("run", 0) == args.plan_run
        and row.get("profile", "main") == args.profile
        and row.get("variant", "neutral") == args.variant
    ]
    write_json(args.output, compare(plans, references, args.alpha, args.margin))


if __name__ == "__main__":
    main()
