import math
from collections import Counter, defaultdict
from statistics import mean, stdev


METRICS = (
    "eta_U",
    "eta_E",
    "regret_U",
    "regret_E",
    "V_U",
    "V_E",
    "V_oracle",
    "V_random",
    "base_accuracy",
)


def _benchmark(dataset):
    dataset = str(dataset).lower().replace("-", "_").replace(" ", "_")
    if dataset.startswith("hle"):
        return "hle"
    return {
        "aime2024": "aime",
        "aime2025": "aime",
        "aime_2024": "aime",
        "aime_2025": "aime",
        "gpqa": "gpqa_diamond",
        "lcb": "livecodebench",
    }.get(dataset, dataset)


def _values(rows, metric):
    return [
        row[metric]
        for row in rows
        if isinstance(row.get(metric), (int, float)) and math.isfinite(row[metric])
    ]


def aggregate_scores(rows):
    groups = defaultdict(list)
    seen = set()
    for index, row in enumerate(rows):
        run = row.get("run", 0)
        key = (
            row.get("profile", "main" if run == 0 else "repeated"),
            row.get("variant", row.get("prompt_variant", "neutral")),
            row.get("configuration", ""),
            _benchmark(row.get("dataset", "")),
            row.get("alpha"),
        )
        identity = key + (run, row.get("category"), row.get("pool_id", ("row", index)))
        if identity in seen:
            raise ValueError(f"Duplicate pool score: {identity}")
        seen.add(identity)
        groups[key].append(row)
    cells = []
    for key, records in sorted(
        groups.items(), key=lambda item: tuple(str(v) for v in item[0])
    ):
        profile, variant, configuration, dataset, alpha = key
        by_run = defaultdict(list)
        for row in records:
            by_run[row.get("run", 0)].append(row)
        runs = []
        for run, entries in sorted(by_run.items(), key=lambda item: str(item[0])):
            scored = [row for row in entries if row.get("status", "ok") == "ok"]
            estimate = {"run": run, "n_pools": len(scored), "n_records": len(entries)}
            for metric in METRICS:
                values = _values(scored, metric)
                estimate[metric] = mean(values) if values else None
                estimate[f"{metric}_n_pools"] = len(values)
            runs.append(estimate)
        available = [run for run in runs if run["n_pools"] > 0]
        cell = {
            "profile": profile,
            "variant": variant,
            "configuration": configuration,
            "dataset": dataset,
            "alpha": alpha,
            "n_runs": len(available),
            "run_ids": [run["run"] for run in available],
            "n_records": len(records),
            "n_scored": sum(run["n_pools"] for run in runs),
            "status_counts": dict(Counter(row.get("status", "ok") for row in records)),
            "runs": runs,
        }
        for field in ("model", "mode"):
            if field in records[0]:
                cell[field] = records[0][field]
        for metric in METRICS:
            values = _values(available, metric)
            cell[metric] = mean(values) if values else None
            cell[f"{metric}_sd"] = stdev(values) if len(values) >= 2 else None
            cell[f"{metric}_n_runs"] = len(values)
        cells.append(cell)
    headline_groups = defaultdict(list)
    for cell in cells:
        headline_groups[(cell["profile"], cell["variant"], cell["alpha"])].append(cell)
    overall = []
    for key, entries in sorted(
        headline_groups.items(), key=lambda item: tuple(str(v) for v in item[0])
    ):
        profile, variant, alpha = key
        available = [cell for cell in entries if cell["n_runs"] > 0]
        result = {
            "profile": profile,
            "variant": variant,
            "alpha": alpha,
            "n_cells": len(available),
            "n_configurations": len({cell["configuration"] for cell in available}),
            "n_benchmarks": len({cell["dataset"] for cell in available}),
        }
        for metric in METRICS:
            values = _values(available, metric)
            result[metric] = mean(values) if values else None
            result[f"{metric}_n_cells"] = len(values)
        overall.append(result)
    return {"cells": cells, "overall": overall}
