import argparse
from collections import defaultdict
from itertools import combinations
from copy import deepcopy
from pathlib import Path
from statistics import mean

from triage_eval.io import read_json, read_records, write_json
from triage_eval.planning import load_configuration, run_planning
from triage_eval.scoring import score_pool

if __package__:
    from .repeated_planning import kendall_tau_b
else:
    from repeated_planning import kendall_tau_b


def ablation_configuration(configuration):
    identifier = configuration["id"]
    if not identifier.startswith(("kimi-k2.5-", "qwen3.5-27b-")):
        raise ValueError(
            "The prompt ablation uses Kimi K2.5 and Qwen 3.5 27B configurations"
        )
    selected = deepcopy(configuration)
    model_id = (
        "moonshotai/kimi-k2.5:nitro"
        if identifier.startswith("kimi")
        else "qwen/qwen3.5-27b:nitro"
    )
    selected.update(
        provider="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        model_id=model_id,
        planner_max_tokens=65536,
        temperature=0.0,
        planner_temperature=0.0,
        reasoning_effort="high" if selected["mode"] == "extended" else "none",
    )
    selected.pop("disable_reasoning", None)
    return selected


def analyze(scores):
    metrics = ("eta_U", "eta_E", "regret_U", "regret_E")
    groups = defaultdict(list)
    for row in scores:
        if row.get("status") == "ok":
            key = (
                row.get("profile", "main"),
                row["configuration"],
                float(row["alpha"]),
                row.get("variant", "neutral"),
                row.get("run", 0),
                row["dataset"],
                row.get("category"),
            )
            groups[key].append(row)
    splits = []
    run_groups = defaultdict(list)
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        profile, configuration, alpha, variant, run_id, dataset, category = key
        result = {
            "profile": profile,
            "configuration": configuration,
            "alpha": alpha,
            "variant": variant,
            "run": run_id,
            "dataset": dataset,
            "category": category,
            "n_pools": len(rows),
        }
        for metric in metrics:
            values = [row[metric] for row in rows if row.get(metric) is not None]
            result[metric] = mean(values) if values else None
        splits.append(result)
        run_groups[(profile, configuration, alpha, variant, run_id)].append(result)
    variant_groups = defaultdict(list)
    for (profile, configuration, alpha, variant, run_id), rows in run_groups.items():
        result = {"run": run_id, "n_splits": len(rows)}
        for metric in metrics:
            values = [row[metric] for row in rows if row[metric] is not None]
            result[metric] = mean(values) if values else None
        variant_groups[(profile, configuration, alpha, variant)].append(result)
    variants = []
    by_configuration = defaultdict(list)
    rankings = defaultdict(lambda: defaultdict(dict))
    for (profile, configuration, alpha, variant), runs in sorted(
        variant_groups.items()
    ):
        result = {
            "profile": profile,
            "configuration": configuration,
            "alpha": alpha,
            "variant": variant,
            "runs": runs,
        }
        for metric in metrics:
            values = [row[metric] for row in runs if row[metric] is not None]
            result[metric] = mean(values) if values else None
            if result[metric] is not None:
                rankings[(profile, alpha, metric)][variant][configuration] = result[
                    metric
                ]
        variants.append(result)
        by_configuration[(profile, configuration, alpha)].append(result)
    ranges = []
    for (profile, configuration, alpha), rows in sorted(by_configuration.items()):
        result = {
            "profile": profile,
            "configuration": configuration,
            "alpha": alpha,
            "n_variants": len(rows),
        }
        for metric in metrics:
            values = [row[metric] for row in rows if row[metric] is not None]
            result[metric] = max(values) - min(values) if values else None
        ranges.append(result)
    correlations = []
    for (profile, alpha, metric), comparisons in sorted(rankings.items()):
        for first, second in combinations(sorted(comparisons), 2):
            left, right = comparisons[first], comparisons[second]
            correlations.append(
                {
                    "profile": profile,
                    "alpha": alpha,
                    "metric": metric,
                    "variant_a": first,
                    "variant_b": second,
                    "n_configurations": len(set(left) & set(right)),
                    "tau_b": kendall_tau_b(left, right),
                }
            )
    return {
        "splits": splits,
        "variants": variants,
        "ranges": ranges,
        "ranking_correlations": correlations,
    }


def run(
    problems, references, configuration, settings, alphas=(0.25, 0.5, 0.75), run_id=0
):
    configuration = ablation_configuration(configuration)
    matched = [row for row in references if row["configuration"] == configuration["id"]]
    plans = []
    scores = []
    for variant in ("neutral", "agent", "exam"):
        current = run_planning(
            problems,
            matched,
            configuration,
            settings,
            run=run_id,
            alphas=list(alphas),
            variant=variant,
        )
        current_scores = []
        for plan in current:
            plan["variant"] = variant
            scored = score_pool(plan, matched)
            scored["variant"] = variant
            current_scores.append(scored)
        plans.extend(current)
        scores.extend(current_scores)
    return {"records": plans, "scores": scores, "summary": analyze(scores)}


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--problems")
    source.add_argument("--scores", nargs="+")
    parser.add_argument("--references")
    parser.add_argument("--configuration")
    parser.add_argument(
        "--settings",
        default=str(Path(__file__).resolve().parents[1] / "configs/main.json"),
    )
    parser.add_argument("--profile", choices=("main", "repeated"), default="main")
    parser.add_argument("--reference-run", type=int, default=0)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.scores:
        scores = []
        for path in args.scores:
            value = read_json(path) if Path(path).suffix.lower() == ".json" else None
            scores.extend(
                value["scores"]
                if isinstance(value, dict) and isinstance(value.get("scores"), list)
                else read_records(path)
            )
        write_json(args.output, analyze(scores))
        return
    if not args.references or not args.configuration:
        parser.error("--references and --configuration are required with --problems")
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
    )
    write_json(args.output, result)


if __name__ == "__main__":
    main()
