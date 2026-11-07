import argparse
import math
from pathlib import Path

from triage_eval.grading import grade
from triage_eval.io import read_json, read_records, write_json
from triage_eval.parsing import parse_plan
from triage_eval.planning import (
    build_solver_prompt,
    load_configuration,
    request_completion,
)
from triage_eval.scoring import get_budget, score_pool

if __package__:
    from .budget_aware import (
        build_budget_aware_prompt,
        is_budget_aware_configuration,
        index_problems,
        execution_settings,
    )
else:
    from budget_aware import (
        build_budget_aware_prompt,
        is_budget_aware_configuration,
        index_problems,
        execution_settings,
    )


VARIANT_A_RULES = "You are given a set of competition mathematics problems to attempt under a fixed output-token budget. You do not have to attempt every problem. Deciding which problems to attempt and in what order is part of the task.\n\nRules\n-----\n- Each problem is worth the number of points listed next to it. Your goal is to maximize total points earned.\n- A point reward is earned only for a problem answered correctly.\n- Attempting a problem consumes output tokens whether or not the answer is correct.\n- Your budget covers every token you generate while solving, including any internal reasoning, drafts, or scratch work.\n- Each problem is attempted in an independent model session. The tokens you allocate to a problem are spent entirely within that single session, with no context, reasoning, or scratchpad carried over between problems.\n- Problems you do not list in your plan are skipped and consume no tokens.\n- Problems are attempted in the order you list them. If your budget is exhausted before a problem is reached, that problem — and every problem after it in your plan — scores zero."


def build_replan_prompt(
    B_local,
    remaining,
    executed_history,
    last_executed,
    remaining_in_plan_items,
    unchosen_problems,
    problem_lookup,
):
    lines = []
    lines.append(VARIANT_A_RULES)
    lines.append("")
    lines.append(f"ORIGINAL TOTAL BUDGET: {B_local} output tokens")
    lines.append(f"REMAINING BUDGET:      {remaining} output tokens")
    lines.append("")
    lines.append("Execution history so far")
    lines.append("------------------------")
    if not executed_history:
        lines.append("(none yet)")
    else:
        for h in executed_history:
            label = "CORRECT" if h["y_i"] == 1 else "INCORRECT"
            lines.append(
                f"  - [{h['pid']}] you predicted {h['a_i']} tok, actually used {h['c_i']} tok, answer was {label}"
            )
    lines.append("")
    if last_executed is not None:
        label = "CORRECT" if last_executed["y_i"] == 1 else "INCORRECT"
        lines.append("Most recently attempted")
        lines.append("-----------------------")
        lines.append(f"  Problem:           [{last_executed['pid']}]")
        lines.append(f"  Tokens predicted:  {last_executed['a_i']}")
        lines.append(f"  Tokens spent:      {last_executed['c_i']}")
        lines.append(f"  Result:            {label}")
        lines.append(f"  Remaining budget:  {remaining}")
        lines.append("")
    lines.append("Remaining problems currently in your plan (in priority order)")
    lines.append("-------------------------------------------------------------")
    if not remaining_in_plan_items:
        lines.append("(none — your current plan is empty)")
    else:
        for it in remaining_in_plan_items:
            pid = it["id"]
            if pid not in problem_lookup:
                continue
            lines.append("")
            lines.append(
                f"  [{pid}] (points: 1, current allocation: {it.get('tokens', 0)} tok)"
            )
            lines.append("  " + problem_lookup[pid]["text"].replace("\n", "\n  "))
    lines.append("")
    lines.append("Problems you did not originally include (still available)")
    lines.append("---------------------------------------------------------")
    if not unchosen_problems:
        lines.append("(none)")
    else:
        for p in unchosen_problems:
            lines.append("")
            lines.append(f"  [{p['id']}] (points: 1)")
            lines.append("  " + p["text"].replace("\n", "\n  "))
    lines.append("")
    lines.append("Your task")
    lines.append("---------")
    lines.append(
        "You may revise your plan for the remaining budget, or keep it as-is. Include only problems you still intend to attempt, in priority order. Do not include any problem you have already attempted (they cannot be retried). An empty array commits no further tokens. The independent-session rule still applies to any remaining attempts."
    )
    lines.append("")
    lines.append(
        f'Return a single JSON array, ordered by solve priority (the first entry is solved first). An empty array is valid if you choose to attempt no further problems.\n\nSchema:\n\n  [\n    {{"id":     "<problem id, exactly as listed>",\n     "tokens": <your estimate of output tokens required to solve this problem correctly>,\n     "reason": "<one short sentence on why this problem is in the plan at this position and allocation>"}}\n    ,\n    ...\n  ]\n\nYour total estimated tokens across all entries should not exceed {remaining}.\n\nRespond with only the JSON array — no surrounding prose, no markdown code fences.'
    )
    return "\n".join(lines)


def run_pool(record, references, problems, configuration, settings):
    configuration = dict(
        configuration,
        solver_max_tokens=4096,
        planner_max_tokens=16384,
        temperature=0.0,
        solver_temperature=0.0,
        planner_temperature=0.0,
    )
    fixed = score_pool(record, references)
    output = {
        key: record.get(key)
        for key in ("configuration", "dataset", "category", "pool_id", "alpha", "run")
    }
    output.update(
        {
            "budget": get_budget(record),
            "problem_ids": list(record["problem_ids"]),
            "initial_plan": record.get("parsed_plan", {}),
            "configuration_settings": configuration,
            "execution_settings": execution_settings(settings),
            "one_shot_value": fixed.get("V_U"),
            "reference_oracle": fixed.get("V_oracle"),
        }
    )
    if fixed["status"] != "ok":
        output.update({"status": fixed["status"], "turns": []})
        return output
    identifiers = list(record["problem_ids"])
    indexed = index_problems(problems)
    lookup = {
        identifier: indexed.get(
            (record["configuration"], record["dataset"], identifier),
            indexed.get((None, record["dataset"], identifier)),
        )
        for identifier in identifiers
    }
    lookup = {
        identifier: problem
        for identifier, problem in lookup.items()
        if problem is not None
    }
    if set(identifiers) != set(lookup):
        raise ValueError("All problems in a feedback pool must be provided")
    reference_lookup = {
        str(row["id"]): row
        for row in references
        if row.get("configuration", record["configuration"]) == record["configuration"]
        and row.get("dataset", record["dataset"]) == record["dataset"]
    }
    for identifier, problem in lookup.items():
        reference = reference_lookup.get(identifier)
        if (
            reference is not None
            and "text" in reference
            and (
                problem["text"] != reference["text"]
                or problem.get("metadata", {}) != reference.get("metadata", {})
            )
        ):
            raise ValueError(
                "Problem text and grading metadata must match the selected reference"
            )
    current = list(record["parsed_plan"]["plan"])
    remaining = get_budget(record)
    history = []
    turns = []
    termination = "empty_plan"
    solver_input = solver_output = replan_input = replan_output = 0
    while current and remaining > 0 and len(history) < len(identifiers):
        item = current.pop(0)
        identifier = item["id"]
        executed = {row["pid"] for row in history}
        if identifier in executed or identifier not in lookup:
            continue
        problem = lookup[identifier]
        prompt = build_budget_aware_prompt(build_solver_prompt(problem), item["tokens"])
        try:
            response = request_completion(
                configuration,
                prompt,
                phase="solver",
                settings=settings,
                dataset="aime",
                run=record.get("run", 0),
            )
        except Exception as error:
            turns.append(
                {
                    "type": "solver_error",
                    "id": identifier,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            termination = "solver_error"
            break
        cost = int(response["output_tokens"])
        if cost < 0:
            raise ValueError("Output-token cost cannot be negative")
        correctness = grade(problem, response["text"])
        history.append(
            {
                "pid": identifier,
                "a_i": item["tokens"],
                "c_i": cost,
                "y_i": int(bool(correctness)),
            }
        )
        remaining -= cost
        solver_input += int(response["input_tokens"])
        solver_output += cost
        turns.append(
            {
                "type": "execution",
                "id": identifier,
                "allocation": item["tokens"],
                "y": correctness,
                "c": cost,
                "remaining": remaining,
                "response": response["text"],
                "reasoning": response.get("reasoning", ""),
                "input_tokens": response["input_tokens"],
                "output_tokens": cost,
            }
        )
        if remaining <= 0:
            termination = "budget_exhausted"
            break
        executed = {row["pid"] for row in history}
        if executed >= set(identifiers):
            termination = "all_problems_executed"
            break
        unchosen = [
            lookup[pid]
            for pid in identifiers
            if pid not in executed and pid not in {entry["id"] for entry in current}
        ]
        prompt = build_replan_prompt(
            get_budget(record),
            remaining,
            history,
            history[-1],
            current,
            unchosen,
            lookup,
        )
        try:
            response = request_completion(
                configuration,
                prompt,
                phase="planner",
                settings=settings,
                dataset="aime",
                run=record.get("run", 0),
            )
        except Exception as error:
            turns.append(
                {"type": "replan_error", "error": f"{type(error).__name__}: {error}"}
            )
            termination = "empty_plan" if not current else "plan_exhausted"
            continue
        replan_input += int(response["input_tokens"])
        replan_output += int(response["output_tokens"])
        parsed = parse_plan(response["text"], identifiers)
        turns.append(
            {
                "type": "replan",
                "remaining": remaining,
                "response": response["text"],
                "reasoning": response.get("reasoning", ""),
                "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
                "parsed_plan": parsed,
            }
        )
        if parsed["ok"]:
            current = [entry for entry in parsed["plan"] if entry["id"] not in executed]
            termination = "empty_replan" if not current else "plan_exhausted"
    output.update(
        {
            "status": "error" if termination == "solver_error" else "completed",
            "feedback_value": sum(row["y_i"] for row in history),
            "n_executed": len(history),
            "remaining": remaining,
            "solver_input_tokens": solver_input,
            "solver_output_tokens": solver_output,
            "replan_input_tokens": replan_input,
            "replan_output_tokens": replan_output,
            "overhead": (replan_input + replan_output) / solver_output
            if solver_output
            else None,
            "termination": termination,
            "turns": turns,
        }
    )
    return output


def run(
    plans,
    references,
    problems,
    settings,
    profile="main",
    output=None,
    configurations=None,
    force=False,
    plan_run=0,
):
    records = (
        read_records(output) if output and Path(output).exists() and not force else []
    )
    if any(
        row.get("profile", "main") != profile or row.get("run", 0) != plan_run
        for row in records
    ):
        raise ValueError(
            "Use separate output files for different inference profiles and planning runs"
        )
    completed = {
        (row["configuration"], row["pool_id"], row.get("run", 0)): row
        for row in records
        if row.get("status") == "completed" and row.get("termination") != "solver_error"
    }
    for plan in plans:
        if (
            plan.get("profile", "main") != profile
            or plan.get("variant", "neutral") != "neutral"
            or plan.get("run", 0) != plan_run
        ):
            continue
        if plan["dataset"] != "aime" or not math.isclose(float(plan["alpha"]), 1.0):
            continue
        if configurations and plan["configuration"] not in configurations:
            continue
        key = (plan["configuration"], plan["pool_id"], plan.get("run", 0))
        configuration = load_configuration(plan["configuration"], profile)
        if not is_budget_aware_configuration(configuration):
            continue
        if key in completed:
            previous = completed[key]
            expected = dict(
                configuration,
                solver_max_tokens=4096,
                planner_max_tokens=16384,
                temperature=0.0,
                solver_temperature=0.0,
                planner_temperature=0.0,
            )
            if (
                previous.get("initial_plan") != plan.get("parsed_plan")
                or previous.get("problem_ids") != plan["problem_ids"]
                or previous.get("budget") != get_budget(plan)
                or previous.get("configuration_settings") != expected
                or previous.get("execution_settings") != execution_settings(settings)
            ):
                raise ValueError(
                    "Existing feedback run does not match this plan and configuration"
                )
            continue
        row = run_pool(plan, references, problems, configuration, settings)
        row["profile"] = profile
        records = [
            existing
            for existing in records
            if (existing["configuration"], existing["pool_id"], existing.get("run", 0))
            != key
        ]
        records.append(row)
        if output:
            write_json(output, {"records": records})
    result = {"records": records}
    if output:
        write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--configuration", nargs="+")
    parser.add_argument(
        "--settings",
        default=str(Path(__file__).resolve().parents[1] / "configs/main.json"),
    )
    parser.add_argument("--profile", choices=("main", "repeated"), default="main")
    parser.add_argument("--reference-run", type=int, default=0)
    parser.add_argument("--plan-run", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    references = [
        row
        for row in read_records(args.references)
        if row.get("run", 0) == args.reference_run
    ]
    run(
        read_records(args.plans),
        references,
        read_records(args.problems),
        read_json(args.settings),
        args.profile,
        args.output,
        args.configuration,
        args.force,
        args.plan_run,
    )


if __name__ == "__main__":
    main()
