import argparse
import math
from collections import defaultdict
from itertools import combinations
from statistics import mean

from triage_eval.aggregation import aggregate_scores
from triage_eval.io import read_records, write_json
from triage_eval.scoring import score_pool


def kendall_tau_b(first, second):
    identifiers = sorted(set(first) & set(second))
    concordant = discordant = ties_first = ties_second = 0
    for left, right in combinations(identifiers, 2):
        x = first[left] - first[right]
        y = second[left] - second[right]
        if x == 0 and y == 0:
            continue
        if x == 0:
            ties_first += 1
        elif y == 0:
            ties_second += 1
        elif x * y > 0:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_first) * (concordant + discordant + ties_second)
    )
    return (concordant - discordant) / denominator if denominator else None


def analyze(scores):
    summary = aggregate_scores(scores)
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in scores:
        if row.get("status") != "ok":
            continue
        for metric in ("eta_U", "eta_E", "regret_U", "regret_E"):
            if row.get(metric) is not None:
                groups[
                    (
                        row.get("profile", "main"),
                        row.get("variant", "neutral"),
                        float(row["alpha"]),
                        metric,
                    )
                ][row["run"]][(row["configuration"], row["dataset"])].append(
                    row[metric]
                )
    rankings = []
    for (profile, variant, alpha, metric), runs in sorted(groups.items()):
        model_means = {}
        for run, cells in runs.items():
            configurations = defaultdict(list)
            for (configuration, dataset), values in cells.items():
                configurations[configuration].append(mean(values))
            model_means[run] = {
                key: mean(values) for key, values in configurations.items()
            }
        for first, second in combinations(sorted(model_means, key=str), 2):
            left, right = model_means[first], model_means[second]
            rankings.append(
                {
                    "profile": profile,
                    "variant": variant,
                    "alpha": alpha,
                    "metric": metric,
                    "run_a": first,
                    "run_b": second,
                    "n_configurations": len(set(left) & set(right)),
                    "tau_b": kendall_tau_b(left, right),
                }
            )
    summary["ranking_correlations"] = rankings
    return summary


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scores")
    source.add_argument("--plans")
    parser.add_argument("--references")
    parser.add_argument("--reference-run", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.scores:
        scores = read_records(args.scores)
    else:
        if not args.references:
            parser.error("--references is required with --plans")
        references = [
            row
            for row in read_records(args.references)
            if row.get("run", 0) == args.reference_run
        ]
        scores = [score_pool(plan, references) for plan in read_records(args.plans)]
    write_json(args.output, analyze(scores))


if __name__ == "__main__":
    main()
