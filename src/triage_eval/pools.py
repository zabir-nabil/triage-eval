import hashlib
import random
from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_FLOOR


HLE_POOL_KEYS = {
    "Computer Science/AI": "hle_cs_ai",
    "Physics": "hle_physics",
    "Humanities/Social Science": "hle_humanities",
    "Other": "hle_other",
    "Engineering": "hle_engineering",
}


def build_chunks(problem_ids, dataset_name, chunk_size=30, min_chunk=10):
    if chunk_size < 1 or not 0 <= min_chunk <= chunk_size:
        raise ValueError("Invalid pool size or minimum tail size")
    ids = list(problem_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Problem IDs must be unique within a dataset split")
    digest = hashlib.sha256(f"{dataset_name}|triage_v1".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:4], "big"))
    rng.shuffle(ids)
    chunks = [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]
    if len(chunks) > 1 and len(chunks[-1]) < min_chunk:
        chunks[-2].extend(chunks.pop())
    return chunks


def build_pools(problems, pool_size=30, min_tail=10):
    groups = defaultdict(list)
    for problem in problems:
        dataset = problem["dataset"]
        category = problem.get("category") if dataset == "hle" else None
        if dataset == "hle" and category not in HLE_POOL_KEYS:
            raise ValueError(f"Unsupported HLE category: {category}")
        groups[(dataset, category)].append(problem)
    pools = []
    for (dataset, category), rows in groups.items():
        seed_key = HLE_POOL_KEYS[category] if dataset == "hle" else dataset
        by_id = {str(row["id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"Duplicate problem ID in {seed_key}")
        chunks = build_chunks(list(by_id), seed_key, pool_size, min_tail)
        for index, ids in enumerate(chunks):
            pools.append(
                {
                    "pool_id": f"{seed_key}_{index:03d}",
                    "dataset": dataset,
                    "category": category,
                    "problem_ids": ids,
                    "problems": [by_id[pid] for pid in ids],
                }
            )
    return pools


def calculate_budget(costs, alpha):
    try:
        fraction = Decimal(str(alpha))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("Budget ratio must be a number in (0, 1]") from error
    if not fraction.is_finite() or not Decimal(0) < fraction <= Decimal(1):
        raise ValueError("Budget ratio must be a number in (0, 1]")
    values = costs.values() if hasattr(costs, "values") else costs
    total = 0
    for value in values:
        if isinstance(value, dict):
            value = value.get("c", value.get("cost"))
        if value is None:
            continue
        if isinstance(value, bool):
            raise ValueError("Reference token costs must be integers")
        try:
            numeric = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as error:
            raise ValueError("Reference token costs must be integers") from error
        if (
            not numeric.is_finite()
            or numeric < 0
            or numeric != numeric.to_integral_value()
        ):
            raise ValueError("Reference token costs must be non-negative integers")
        total += int(numeric)
    return int((fraction * total).to_integral_value(rounding=ROUND_FLOOR))
