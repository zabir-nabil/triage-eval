import json
import math
import random
from collections.abc import Mapping

from .parsing import parse_plan


def get_budget(plan):
    value = plan.get("budget", plan.get("B_local"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("A plan must contain an integer budget")
    if not math.isfinite(value) or value < 0 or int(value) != value:
        raise ValueError("A plan budget must be a non-negative integer")
    return int(value)


def _dataset_name(value):
    value = str(value).lower().replace("-", "_").replace(" ", "_")
    if value.startswith("hle"):
        return "hle"
    return {
        "aime_2024": "aime",
        "aime_2025": "aime",
        "aime2024": "aime",
        "aime2025": "aime",
        "gpqa": "gpqa_diamond",
        "lcb": "livecodebench",
    }.get(value, value)


def _reference_sets(plan, references):
    if isinstance(references, Mapping):
        source = [(str(problem_id), row) for problem_id, row in references.items()]
    else:
        source = [(str(row["id"]), row) for row in references]
    configuration = plan.get("configuration")
    dataset = plan.get("dataset")
    lookup = {}
    for problem_id, row in source:
        if not isinstance(row, Mapping):
            raise ValueError("Reference records must be objects")
        if (
            configuration is not None
            and row.get("configuration", configuration) != configuration
        ):
            continue
        if dataset is not None and _dataset_name(
            row.get("dataset", dataset)
        ) != _dataset_name(dataset):
            continue
        if problem_id in lookup:
            raise ValueError(
                f"Duplicate reference id {problem_id}; select one reference run"
            )
        lookup[problem_id] = row
    pool_ids = list(
        dict.fromkeys(str(problem_id) for problem_id in plan.get("problem_ids", []))
    )
    matched = {
        problem_id: lookup[problem_id]
        for problem_id in pool_ids
        if problem_id in lookup
    }
    gradeable = {}
    for problem_id, row in matched.items():
        correctness = row.get("y")
        cost = row.get("c")
        if (
            correctness not in (0, 1)
            or not isinstance(cost, (int, float))
            or isinstance(cost, bool)
        ):
            continue
        if not math.isfinite(cost) or cost <= 0 or int(cost) != cost:
            continue
        gradeable[problem_id] = {**row, "y": int(correctness), "c": int(cost)}
    return matched, gradeable


def matched_problems(plan, references):
    matched, gradeable = _reference_sets(plan, references)
    if not matched or len(gradeable) < 0.5 * len(matched):
        return None
    return gradeable


def execute_plan_U(plan_order, problems, budget):
    remaining = budget
    value = 0
    for problem_id in plan_order:
        problem = problems.get(problem_id)
        if problem is None or problem.get("y") is None or problem.get("c") is None:
            continue
        cost = problem["c"]
        if cost > remaining:
            break
        remaining -= cost
        value += problem["y"]
    return value


def execute_plan_E(plan_items, problems, budget):
    remaining = budget
    value = 0
    for item in plan_items:
        problem_id = item.get("id") or item.get("problem_id") or ""
        allocation = item.get("tokens", item.get("estimated_tokens", 0))
        if allocation <= 0:
            continue
        if allocation > remaining:
            break
        remaining -= allocation
        problem = problems.get(problem_id)
        if problem is None or problem.get("y") is None or problem.get("c") is None:
            continue
        if allocation >= problem["c"] and problem["y"] == 1:
            value += 1
    return value


def compute_oracle(problems, budget):
    costs = sorted(
        problem["c"]
        for problem in problems.values()
        if problem.get("y") == 1 and problem.get("c") is not None and problem["c"] > 0
    )
    remaining = budget
    value = 0
    for cost in costs:
        if cost > remaining:
            break
        remaining -= cost
        value += 1
    return value


def compute_random(problems, budget, n_samples=1000, rng_seed=42):
    if n_samples < 1 or int(n_samples) != n_samples:
        raise ValueError(
            "The random reference requires a positive integer sample count"
        )
    rng = random.Random(rng_seed)
    problem_ids = [
        problem_id
        for problem_id, problem in problems.items()
        if problem.get("y") is not None
        and problem.get("c") is not None
        and problem["c"] > 0
    ]
    if not problem_ids:
        return 0.0
    total = 0
    for _ in range(int(n_samples)):
        order = problem_ids.copy()
        rng.shuffle(order)
        total += execute_plan_U(order, problems, budget)
    return total / n_samples


def compute_eta(V_model, V_random, V_oracle):
    denominator = V_oracle - V_random
    if denominator <= 0:
        return 1.0 if V_model >= V_oracle else 0.0
    return (V_model - V_random) / denominator


def compute_regret(V_model, V_oracle):
    if V_oracle <= 0:
        return None
    return max(0.0, (V_oracle - V_model) / V_oracle)


def score_pool(plan, references, n_samples=1000, rng_seed=42):
    budget = get_budget(plan)
    alpha = plan.get("alpha")
    if alpha is not None and not 0 < float(alpha) <= 1:
        raise ValueError("alpha must be in (0, 1]")
    run = plan.get("run", 0)
    result = {
        key: plan[key]
        for key in (
            "configuration",
            "dataset",
            "category",
            "pool_id",
            "alpha",
            "run",
            "model",
            "mode",
        )
        if key in plan
    }
    result.update(
        budget=budget,
        run=run,
        profile=plan.get("profile", "main" if run == 0 else "repeated"),
        variant=plan.get("variant", plan.get("prompt_variant", "neutral")),
        status="unparseable",
        n_matched=0,
        n_gradeable=0,
        n_solvable=0,
        n_plan_items=0,
        n_plan_gradeable=0,
        base_accuracy=None,
        V_U=None,
        V_E=None,
        V_oracle=None,
        V_random=None,
        eta_U=None,
        eta_E=None,
        regret_U=None,
        regret_E=None,
    )
    parsed = plan.get("parsed_plan")
    if isinstance(parsed, Mapping):
        if not parsed.get("ok") or not isinstance(parsed.get("plan"), list):
            return result
        parsed = parse_plan(
            json.dumps(parsed["plan"]), plan.get("problem_ids", []), recovery=False
        )
    elif isinstance(plan.get("plan"), list):
        parsed = parse_plan(
            json.dumps(plan["plan"]), plan.get("problem_ids", []), recovery=False
        )
    else:
        parsed = parse_plan(plan.get("response", ""), plan.get("problem_ids", []))
    if not parsed["ok"]:
        return result
    items = parsed["plan"]
    result["n_plan_items"] = len(items)
    matched, problems = _reference_sets(plan, references)
    result.update(n_matched=len(matched), n_gradeable=len(problems))
    if not matched:
        result["status"] = "no_reference"
        return result
    if len(problems) < 0.5 * len(matched):
        result["status"] = "insufficient_reference"
        return result
    selected = [item for item in items if item["id"] in problems]
    V_U = execute_plan_U([item["id"] for item in selected], problems, budget)
    V_E = execute_plan_E(selected, problems, budget)
    V_oracle = compute_oracle(problems, budget)
    V_random = compute_random(problems, budget, n_samples, rng_seed)
    n_solvable = sum(problem["y"] for problem in problems.values())
    full_budget = alpha is not None and math.isclose(
        float(alpha), 1.0, rel_tol=0, abs_tol=1e-12
    )
    result.update(
        status="ok",
        n_solvable=n_solvable,
        n_plan_gradeable=len(selected),
        base_accuracy=n_solvable / len(problems) if problems else None,
        V_U=V_U,
        V_E=V_E,
        V_oracle=V_oracle,
        V_random=V_random,
        eta_U=None if full_budget else compute_eta(V_U, V_random, V_oracle),
        eta_E=None if full_budget else compute_eta(V_E, V_random, V_oracle),
        regret_U=compute_regret(V_U, V_oracle),
        regret_E=compute_regret(V_E, V_oracle),
    )
    return result
