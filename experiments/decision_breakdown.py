import argparse
import math
from collections import Counter, defaultdict
from statistics import mean

from triage_eval.io import read_records, write_json
from triage_eval.scoring import get_budget, matched_problems


def classify(plan, problems, budget, regime):
    remaining = budget
    stopped = False
    outcomes = {}
    for item in plan:
        identifier = item["id"]
        if identifier not in problems:
            continue
        problem = problems[identifier]
        cost = problem["c"] if regime == "U" else item["tokens"]
        if stopped:
            outcomes[identifier] = "unreached"
        elif regime == "E" and cost <= 0:
            outcomes[identifier] = "zero_allocation"
        elif cost > remaining:
            stopped = True
            outcomes[identifier] = "unreached"
        else:
            remaining -= cost
            if regime == "E" and cost < problem["c"]:
                outcomes[identifier] = "underallocated"
            else:
                outcomes[identifier] = (
                    "reached_correct" if problem["y"] else "reached_incorrect"
                )
    for identifier, problem in problems.items():
        if identifier not in outcomes:
            outcomes[identifier] = (
                "omitted_correct" if problem["y"] else "omitted_incorrect"
            )
    return Counter(outcomes.values())


def analyze(plans, references, alpha=0.5):
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
    totals = defaultdict(Counter)
    for record in plans:
        if not math.isclose(float(record["alpha"]), alpha) or not record.get(
            "parsed_plan", {}
        ).get("ok"):
            continue
        problems = matched_problems(record, references)
        if not problems:
            continue
        plan = [
            item for item in record["parsed_plan"]["plan"] if item["id"] in problems
        ]
        for regime in ("U", "E"):
            counts = classify(plan, problems, get_budget(record), regime)
            totals[regime].update(counts)
            row = {
                key: record.get(key)
                for key in ("configuration", "dataset", "pool_id", "run", "alpha")
            }
            row.update(
                {"regime": regime, "n_items": len(problems), "counts": dict(counts)}
            )
            rows.append(row)
    summary = {}
    for regime, counts in totals.items():
        total = sum(counts.values())
        current = [row for row in rows if row["regime"] == regime]
        categories = (
            "reached_correct",
            "reached_incorrect",
            "underallocated",
            "zero_allocation",
            "unreached",
            "omitted_correct",
            "omitted_incorrect",
        )
        summary[regime] = {
            "n_items": total,
            "n_pools": len(current),
            "counts": dict(counts),
            "mean_pool_fractions": {
                key: mean(row["counts"].get(key, 0) / row["n_items"] for row in current)
                for key in categories
            },
        }
    return {"pools": rows, "overall": summary}


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
    write_json(args.output, analyze(plans, references, args.alpha))


if __name__ == "__main__":
    main()
