# Supporting experiments

Run these commands from the repository root after installing the package. Reference files retain problem text and metadata, so they can also be supplied to `--problems`.

## Reference stability

Analyze independently sampled Phase-0 reference measurements.

```sh
python experiments/reference_stability.py --references outputs/repeated_references.json --output outputs/reference_stability.json
```

## Repeated planning

Aggregate scored pools into planning-run estimates, report sample standard deviations, and compare model rankings across runs. 

```sh
python experiments/repeated_planning.py --scores outputs/repeated_scores.json --output outputs/repeated_planning.json
```

## Allocation replacements

Keep the selected items, order, and total budget fixed. Compare declared allocations, measured costs, measured costs plus a margin, and an equal split. Summary efficiencies give equal weight to configurations; the margin is charged against the same budget.

```sh
python experiments/allocation_counterfactual.py --plans outputs/plans.json --references outputs/references.json --alpha 0.5 --margin 0.2 --output outputs/allocation_counterfactual.json
```

## Decision outcomes

Count reached, omitted, under-allocated, and unreached items. The reported outcome fractions average the fractions within individual pools.

```sh
python experiments/decision_breakdown.py --plans outputs/plans.json --references outputs/references.json --alpha 0.5 --output outputs/decision_breakdown.json
```

## Reference-solved pools

Generate new plans using only successful reference completions from AIME, GPQA Diamond, and LiveCodeBench. 

```sh
python experiments/reference_solved_pools.py --problems outputs/references.json --references outputs/references.json --configuration gemini-2.5-flash-standard --alphas 0.5 0.75 --output outputs/reference_solved_gemini_standard.json
```

## Unanswerable items

The main probe uses GPQA-Abstain and MMLU-Math-Abstain and detects injected items that receive no positive allocation. Its four configurations are `gemini-2.5-flash-extended`, `qwen2.5-32b-standard`, `qwen3-32b-extended`, and `gpt-oss-20b-extended`.

The expanded probe uses GPQA-Abstain and detects injected items absent from the plan, including zero-token entries as selected. 

These commands measure reference completions when `--references` is omitted. Reuse the same output path when adding another configuration; completed requests are retained. Main and expanded probes require separate output files.

```sh
python experiments/unanswerable_main.py --configuration gemini-2.5-flash-extended --output outputs/unanswerable_main.json
python experiments/unanswerable_expanded.py --configuration qwen3.5-27b-standard --output outputs/unanswerable_expanded.json
```

Recalculate metrics from saved plans without making model calls:

```sh
python experiments/unanswerable_main.py --plans outputs/unanswerable_main.json --output outputs/unanswerable_main_metrics.json
python experiments/unanswerable_expanded.py --plans outputs/unanswerable_expanded.json --output outputs/unanswerable_expanded_metrics.json
```

## Prompt sensitivity

Generate neutral, agent, and exam plans at budget ratios 0.25, 0.5, and 0.75. The ablation uses Kimi K2.5 and Qwen 3.5 27B through OpenRouter, each in standard and extended mode, with a 65,536-token planning ceiling and temperature zero. Its aggregate gives equal weight to the eight dataset splits, keeping the five HLE categories separate.

```sh
python experiments/prompt_sensitivity.py --problems outputs/references.json --references outputs/references.json --configuration kimi-k2.5-standard --output outputs/prompt_kimi_standard.json
```

The output contains plans in `records`, their scores in `scores`, and the ablation summary. Analyze collected score files with `--scores`; this also computes across-variant ranges and Kendall rank correlations.

```sh
python experiments/prompt_sensitivity.py --scores outputs/prompt_kimi_standard.json --output outputs/prompt_sensitivity.json
```

## Budget-aware re-solving

Re-solve the positive-allocation selections from full-budget plans for standard-mode Gemini, Kimi, and GLM. The solver receives its declared allocation in the prompt; the API output ceiling is unchanged. Compliance is measured from the returned output-token count. LiveCodeBench grading acquires the release_v6 tests; `--lcb-tests` accepts an existing local test database.

```sh
python experiments/budget_aware.py --plans outputs/plans.json --references outputs/references.json --problems outputs/references.json --output outputs/budget_aware.json
```

## Feedback replanning

Use the two full-budget AIME pools for standard-mode Gemini, Kimi, and GLM. Fresh solver output is charged to the solving budget. Replanning input and output are recorded separately and do not reduce that budget. 

```sh
python experiments/feedback_replanning.py --plans outputs/plans.json --references outputs/references.json --problems outputs/references.json --output outputs/feedback_replanning.json
```

The allocation, decision-outcome, budget-aware, and feedback commands select planning run zero by default. Use `--plan-run` to select a different run. `--reference-run` selects the reference measurements and defaults to zero.
