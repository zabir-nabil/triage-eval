import argparse
from collections import defaultdict
from statistics import mean, median, pstdev

from triage_eval.io import read_records, write_json
from triage_eval.pools import build_pools


def coefficient_of_variation(values):
    average = mean(values)
    return pstdev(values) / average if len(values) > 1 and average > 0 else 0.0


def analyze(references, include_main=False):
    profiles = {
        row.get("profile", "main")
        for row in references
        if include_main or str(row.get("run", 0)) != "0"
    }
    if len(profiles) > 1:
        raise ValueError("Select one inference profile for reference stability")
    groups = defaultdict(lambda: defaultdict(dict))
    for row in references:
        if row.get("y") is None or not row.get("c") or row["c"] <= 0:
            continue
        run = row.get("run", 0)
        if not include_main and str(run) == "0":
            continue
        key = (row["configuration"], row["dataset"])
        identifier = str(row["id"])
        if identifier in groups[key][run]:
            raise ValueError(
                f"Duplicate reference for {key}, run {run}, item {identifier}"
            )
        groups[key][run][identifier] = row
    cells = []
    for (configuration, dataset), runs in sorted(groups.items()):
        if len(runs) < 2:
            continue
        identifiers = set.intersection(*(set(rows) for rows in runs.values()))
        if not identifiers:
            continue
        ordered_runs = sorted(runs, key=str)
        accuracies = [
            mean(runs[run][pid]["y"] for pid in identifiers) for run in ordered_runs
        ]
        agreement = mean(
            len({runs[run][pid]["y"] for run in ordered_runs}) == 1
            for pid in identifiers
        )
        cost_variability = [
            coefficient_of_variation([runs[run][pid]["c"] for run in ordered_runs])
            for pid in identifiers
        ]
        problems = [
            {"id": pid, "dataset": dataset, "category": row.get("category")}
            for pid, row in runs[ordered_runs[0]].items()
            if pid in identifiers
        ]
        pools = build_pools(problems)
        budget_variability = [
            coefficient_of_variation(
                [
                    sum(runs[run][pid]["c"] for pid in pool["problem_ids"])
                    for run in ordered_runs
                ]
            )
            for pool in pools
        ]
        cells.append(
            {
                "configuration": configuration,
                "dataset": dataset,
                "n_items": len(identifiers),
                "n_runs": len(ordered_runs),
                "run_ids": ordered_runs,
                "accuracies": accuracies,
                "accuracy_mean": mean(accuracies),
                "accuracy_sd": pstdev(accuracies),
                "label_agreement": agreement,
                "cost_cv": median(cost_variability),
                "cost_cv_median": median(cost_variability),
                "cost_cv_mean": mean(cost_variability),
                "pool_cost_cv": mean(budget_variability)
                if budget_variability
                else None,
            }
        )
    metrics = (
        "accuracy_sd",
        "label_agreement",
        "cost_cv",
        "cost_cv_median",
        "cost_cv_mean",
        "pool_cost_cv",
    )
    overall = {
        name: mean(row[name] for row in cells if row[name] is not None)
        if any(row[name] is not None for row in cells)
        else None
        for name in metrics
    }
    return {"cells": cells, "overall": overall, "n_cells": len(cells)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True)
    parser.add_argument("--include-main", action="store_true")
    parser.add_argument("--profile", choices=("main", "repeated"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    references = read_records(args.references)
    if args.profile:
        references = [
            row for row in references if row.get("profile", "main") == args.profile
        ]
    write_json(args.output, analyze(references, args.include_main))


if __name__ == "__main__":
    main()
