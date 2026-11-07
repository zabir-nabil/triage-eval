TRIAGE
======

Code for “TRIAGE: Evaluating Prospective Metacognitive Control in LLMs under Resource Constraints.”

Installation
------------

Python 3.11 or later is required. Scoring uses the Python standard library.

```sh
uv sync --extra data --extra api
```

For local inference on a Linux GPU worker:

```sh
uv sync --extra data --extra api --extra gpu
```

Copy `.env.example` to `.env` and fill the credentials for the providers being used. Hugging Face access may require accepting the source dataset's terms. AWS credentials can also come from the standard AWS credential chain.

Main evaluation
---------------

Run commands from this directory. This example evaluates GPT-5 Mini in standard mode on AIME:

```sh
uv run --env-file .env python scripts/phase0_reference.py --configuration gpt-5-mini-standard --datasets aime --output outputs/references.json
uv run --env-file .env python scripts/phase1_plan.py --configuration gpt-5-mini-standard --references outputs/references.json --alphas 0.5 1.0 --output outputs/plans.json
uv run python scripts/phase2_score.py --references outputs/references.json --plans outputs/plans.json --output outputs/scores.json
```

`phase0_reference.py` acquires the selected benchmarks, measures independent completions, and records correctness and token costs. `phase1_plan.py` constructs pools and generates plans against those fixed measurements. `phase2_score.py` applies U/E scoring and writes pool scores and aggregate results.

Omit `--datasets` to use all four benchmarks. `--configuration all` selects the 30 model–mode configurations in `configs/models.json`. Omitting `--alphas` uses the four budget ratios in `configs/main.json`. Provider and GPU dependencies must be installed for the selected configurations.

Generation commands checkpoint their outputs and resume completed records. Use a new output path when changing prompts, generation settings, or input problems.

Repeated runs
-------------

Reuse main-run prompts and budgets for repeated planning:

```sh
uv run --env-file .env python scripts/phase1_plan.py --configuration gpt-5-mini-standard --profile repeated --config configs/repeated.json --references outputs/references.json --base-plans outputs/plans.json --output outputs/repeated_plans.json
uv run python scripts/phase2_score.py --references outputs/references.json --plans outputs/repeated_plans.json --output outputs/repeated_scores.json
```

The repeated profile samples three plans at the half and full reference budgets. Main reference outcomes remain fixed. Separate reference repetitions can be generated with `phase0_reference.py --profile repeated --config configs/repeated.json`; these assess solving stability and do not replace the references used to score repeated plans. 

Configuration
-------------

`configs/models.json` contains checkpoint/API identifiers, reasoning modes, provider routes, and generation limits for main and repeated runs. `configs/main.json` contains dataset and pool settings. `configs/repeated.json` contains repetition settings.


Reference records include problem text and grading metadata. They can be supplied through `--problems` to reuse a dataset. JSON and JSONL inputs are supported. Phase-2 JSON output contains `records` and `summary`; unparseable plans and insufficient reference coverage have statuses to detect them.


Supporting experiments and their commands are listed in [experiments/README.md](experiments/README.md). Asset sources and published terms are listed in [ASSET_LICENSES.md](ASSET_LICENSES.md).
