import argparse
import json
from pathlib import Path

from triage_eval.aggregation import aggregate_scores
from triage_eval.io import read_records, write_json, write_jsonl
from triage_eval.scoring import score_pool


def main():
    parser = argparse.ArgumentParser(
        description="Score plans against fixed reference records"
    )
    parser.add_argument("--references", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-run", type=int, default=0)
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    if args.reference_run < 0:
        parser.error("--reference-run must be non-negative")
    if args.random_samples < 1:
        parser.error("--random-samples must be positive")
    references = [
        row
        for row in read_records(args.references)
        if row.get("run", 0) == args.reference_run
    ]
    if not references:
        parser.error("No reference records match --reference-run")
    plans = read_records(args.plans)
    scores = [
        score_pool(
            plan, references, n_samples=args.random_samples, rng_seed=args.random_seed
        )
        for plan in plans
    ]
    summary = aggregate_scores(scores)
    if Path(args.output).suffix.lower() == ".jsonl":
        write_jsonl(args.output, scores)
        print(json.dumps(summary, allow_nan=False))
    else:
        write_json(args.output, {"records": scores, "summary": summary})
        print(json.dumps({"output": str(Path(args.output)), "records": len(scores)}))


if __name__ == "__main__":
    main()
