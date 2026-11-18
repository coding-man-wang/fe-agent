"""Read-only CLI for D5 candidate and A/B evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from agent_core.benchmark.d5_effectiveness import (
    evaluate_d5_effectiveness,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--candidate-run-id")
    mode.add_argument("--ab-report", type=Path)

    parser.add_argument("--task-id", default="TASK-002")
    parser.add_argument("--fe-agent-root", default=".")

    args = parser.parse_args()

    root = (
        Path(args.fe_agent_root).resolve()
        / ".fe-agent"
        / "benchmark"
        / "full_execution"
    )

    if args.candidate_run_id:
        run_id = args.candidate_run_id

        if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
            parser.error("Invalid Run ID")

        if re.fullmatch(r"TASK-[0-9]{3}", args.task_id) is None:
            parser.error("Invalid Task ID")

        path = root / run_id / f"d5_candidate_evidence_{args.task_id}.json"

        result = json.loads(path.read_text(encoding="utf-8-sig"))

        if result.get("task_id") != args.task_id:
            raise SystemExit("Candidate task mismatch")

    else:
        path = args.ab_report.resolve()

        # Read reports only from the expected A/B report directory.
        approved_directory = (root / "ab").resolve()

        if path.parent != approved_directory:
            parser.error("A/B report must be in full_execution/ab")

        report = json.loads(path.read_text(encoding="utf-8-sig"))

        result = evaluate_d5_effectiveness(report)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
