import json
import random
from collections.abc import Mapping


HLE_CATEGORIES = (
    "Computer Science/AI",
    "Engineering",
    "Humanities/Social Science",
    "Other",
    "Physics",
)

MATH_KEYWORDS = (
    "math",
    "algebra",
    "statistics",
    "calculus",
    "geometry",
    "number_theory",
    "arithmetic",
    "probability",
)


def _first(row, *keys, default=None):
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def _hf_load(repo, name=None, split=None, revision=None, **kwargs):
    from datasets import load_dataset as hf_load_dataset

    if revision:
        kwargs["revision"] = revision
    if split:
        kwargs["split"] = split
    if name:
        return hf_load_dataset(repo, name, **kwargs)
    return hf_load_dataset(repo, **kwargs)


def _settings(name, config):
    if not config:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError("Dataset settings must be a mapping")
    datasets = config.get("datasets")
    if isinstance(datasets, Mapping):
        return dict(datasets.get(name, {}))
    return dict(config)


def _problem(pid, dataset, text, metadata, category=None):
    if text is None or not str(text).strip():
        raise ValueError(f"Missing problem text for {pid}")
    return {
        "id": str(pid),
        "dataset": dataset,
        "category": category,
        "text": str(text),
        "metadata": metadata,
    }


def load_aime(settings=None):
    settings = settings or {}
    source_2024 = settings.get("source_2024", "Maxwell-Jia/AIME_2024")
    rows = _hf_load(source_2024, split="train", revision=settings.get("revision_2024"))
    out = []
    for index, row in enumerate(rows):
        pid = _first(row, "ID", default=f"aime2024_{index:03d}")
        text = _first(row, "Problem", "problem", "question")
        answer = _first(row, "Answer", "answer")
        out.append(
            _problem(
                pid,
                "aime",
                text,
                {
                    "answer": str(answer).strip(),
                    "source": "AIME 2024",
                    "source_repo": source_2024,
                },
            )
        )
    source_2025 = settings.get("source_2025")
    candidates = [
        ("MathArena/aime_2025", None, "train"),
        ("MathArena/aime_2025", None, "test"),
        ("yentinglin/aime_2025", None, "train"),
        ("opencompass/AIME2025", "AIME2025-I", "test"),
        ("opencompass/AIME2025", "AIME2025-II", "test"),
    ]
    if source_2025:
        candidates = [
            (
                source_2025,
                settings.get("config_2025"),
                settings.get("split_2025", "train"),
            )
        ]
    failures = []
    for repo, configuration, split in candidates:
        try:
            rows = _hf_load(
                repo,
                name=configuration,
                split=split,
                revision=settings.get("revision_2025"),
            )
            loaded = []
            for index, row in enumerate(rows):
                pid = row.get("problem_idx", row.get("ID", index))
                text = _first(row, "problem", "Problem", "question", "Question")
                answer = _first(row, "answer", "Answer", "target")
                loaded.append(
                    _problem(
                        f"aime2025_{repo.split('/')[-1]}_{pid}",
                        "aime",
                        text,
                        {
                            "answer": str(answer).strip(),
                            "source": f"AIME 2025 ({repo})",
                            "source_repo": repo,
                        },
                    )
                )
            identifiers = {problem["id"] for problem in loaded}
            if not loaded or len(identifiers) != len(loaded):
                raise ValueError("AIME 2025 source must contain unique problem IDs")
            if not source_2025 and len(identifiers) != 30:
                raise ValueError(
                    f"AIME 2025 requires 30 problems; source returned {len(identifiers)}"
                )
            return out + loaded
        except Exception as error:
            if source_2025:
                raise
            failures.append(
                f"{repo}/{configuration or 'default'}/{split}: {type(error).__name__}"
            )
    raise ValueError(
        "Could not load 30 unique AIME 2025 problems: " + "; ".join(failures)
    )


def load_gpqa_diamond(settings=None):
    settings = settings or {}
    source = settings.get("source", "Idavidrein/gpqa")
    rows = _hf_load(
        source, "gpqa_diamond", split="train", revision=settings.get("revision")
    )
    seed = int(settings.get("choice_seed", 42))
    out = []
    for index, row in enumerate(rows):
        choices = [
            (row["Correct Answer"], True),
            (row["Incorrect Answer 1"], False),
            (row["Incorrect Answer 2"], False),
            (row["Incorrect Answer 3"], False),
        ]
        random.Random(seed * 10000 + index).shuffle(choices)
        correct_index = next(i for i, (_, correct) in enumerate(choices) if correct)
        rendered = "\n".join(
            f"({chr(65 + i)}) {text}" for i, (text, _) in enumerate(choices)
        )
        out.append(
            _problem(
                f"gpqa_{index:03d}",
                "gpqa_diamond",
                f"{row['Question']}\n\n{rendered}",
                {
                    "question": row["Question"],
                    "choices": [text for text, _ in choices],
                    "correct_letter": chr(65 + correct_index),
                    "correct_answer_text": row["Correct Answer"],
                    "subdomain": row.get("Subdomain", ""),
                    "high_level_domain": row.get("High-level domain", ""),
                    "shuffle_seed": seed * 10000 + index,
                    "source_repo": source,
                },
            )
        )
    return out


def _livecodebench_rows(settings=None):
    settings = settings or {}
    source = settings.get("source", "livecodebench/code_generation_lite")
    version = settings.get("version", "release_v6")
    if version != "release_v6":
        raise ValueError("The paper protocol uses LiveCodeBench release_v6")
    kwargs = {
        "version_tag": version,
        "split": "test",
        "revision": settings.get("revision"),
    }
    try:
        rows = _hf_load(source, trust_remote_code=True, **kwargs)
    except (TypeError, ValueError):
        rows = _hf_load(source, name="release_latest", **kwargs)
    return sorted(
        (row for row in rows if str(row.get("contest_date", "")).startswith("2025")),
        key=lambda row: (
            str(row.get("contest_date", "")),
            str(row.get("question_id", "")),
        ),
    )


def load_livecodebench(settings=None):
    out = []
    for row in _livecodebench_rows(settings):
        text = row["question_content"]
        starter = row.get("starter_code") or ""
        if starter:
            text += f"\n\nStarter code:\n{starter}"
        out.append(
            _problem(
                row["question_id"],
                "livecodebench",
                text,
                {
                    "title": row.get("question_title", ""),
                    "difficulty": row.get("difficulty", ""),
                    "contest_date": row.get("contest_date", ""),
                    "platform": row.get("platform", ""),
                    "starter_code": starter,
                    "question_content": row["question_content"],
                    "source_repo": "livecodebench/code_generation_lite",
                },
            )
        )
    return out


def load_hle(settings=None):
    settings = settings or {}
    categories = settings.get("categories", HLE_CATEGORIES)
    if (
        isinstance(categories, str)
        or not categories
        or not set(categories).issubset(HLE_CATEGORIES)
    ):
        raise ValueError(
            "HLE categories must be a nonempty subset of the evaluated categories"
        )
    categories = set(categories)
    source = settings.get("source", "cais/hle")
    rows = _hf_load(source, split="test", revision=settings.get("revision"))
    out = []
    for row in rows:
        image = row.get("image")
        if bool(image) and str(image).strip():
            continue
        category = str(row.get("category", "")).strip()
        if category not in categories:
            continue
        answer_type = row.get("answer_type", "short_answer")
        out.append(
            _problem(
                row["id"],
                "hle",
                row.get("question", ""),
                {
                    "answer": str(row.get("answer", "")).strip(),
                    "answer_type": answer_type,
                    "category": category,
                    "raw_subject": row.get("raw_subject", ""),
                    "source_repo": source,
                },
                category,
            )
        )
    out.sort(key=lambda row: row["id"])
    return out


def load_dataset(name, config=None):
    loaders = {
        "aime": load_aime,
        "gpqa_diamond": load_gpqa_diamond,
        "livecodebench": load_livecodebench,
        "hle": load_hle,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}")
    return loaders[name](_settings(name, config))


def load_problems(name, settings=None):
    return load_dataset(name, settings)


def _metadata(value):
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _reference_answers(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            value = decoded if isinstance(decoded, list) else [decoded]
        except ValueError:
            value = [value]
    elif not isinstance(value, (list, tuple)):
        value = [value]
    return [
        str(answer) for answer in value if answer is not None and str(answer).strip()
    ]


def load_unanswerable(dataset="gpqa", settings=None):
    aliases = {"gpqa_abstain": "gpqa", "mmlu_math_abstain": "mmlu_math"}
    dataset = aliases.get(dataset, dataset)
    if dataset not in {"gpqa", "mmlu_math"}:
        raise ValueError(f"Unknown unanswerable-item dataset: {dataset}")
    settings = settings or {}
    data = _hf_load(
        "facebook/AbstentionBench",
        revision=settings.get("revision"),
        trust_remote_code=True,
    )
    rows = data[next(iter(data))] if hasattr(data, "keys") else data
    name = "gpqa_abstain" if dataset == "gpqa" else "mmlu_math_abstain"
    out = []
    for index, row in enumerate(rows):
        meta = _metadata(row.get("metadata_json"))
        if dataset == "gpqa" and set(meta) != {"subdomain"}:
            continue
        if dataset == "mmlu_math":
            subject = str(meta.get("subject", "")).lower()
            if set(meta) != {"subject"} or not any(
                word in subject for word in MATH_KEYWORDS
            ):
                continue
        unanswerable = bool(row.get("should_abstain", False))
        problem = _problem(
            f"{name}_{index}",
            name,
            row.get("question", ""),
            {
                "unanswerable": unanswerable,
                "reference_answers": _reference_answers(row.get("reference_answers")),
                "source_meta": meta,
                "orig_idx": index,
                "source_repo": "facebook/AbstentionBench",
            },
        )
        problem["unanswerable"] = unanswerable
        out.append(problem)
    return out
