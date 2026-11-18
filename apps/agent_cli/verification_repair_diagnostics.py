"""Inspect historical Verification Repair attempts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from agent_core.benchmark.execution_model import (
    FullExecutionRunRecord,
)
from agent_core.benchmark.verification_repair_diagnostics import (
    analyze_repair_outcome,
)


def inspect_saved_repair(
    *,
    run_id: str,
    task_id: str,
    fe_agent_root: Path,
) -> dict:
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError("Invalid Run ID")

    base = Path(fe_agent_root).resolve() / ".fe-agent" / "benchmark" / "full_execution"

    path = base / "runs" / f"{run_id}.json"

    record = FullExecutionRunRecord.model_validate_json(
        path.read_text(encoding="utf-8-sig")
    )

    if record.run_id != run_id:
        raise ValueError("Run ID mismatch")

    task = next(
        (t for t in record.tasks if t.task_id == task_id),
        None,
    )

    if task is None:
        raise ValueError(f"Task not found: {task_id}")

    outcome = analyze_repair_outcome(task.verification_attempts)

    actual_repair_attempts = sum(
        item.kind == "repair" for item in task.verification_attempts
    )

    if actual_repair_attempts != task.verification_repair_count:
        outcome = {
            **outcome,
            "status": "insufficient_evidence",
            "reason": "repair_count_mismatch",
            "all_checks_passed": False,
        }

    if task.verification_attempts:
        last_attempt = task.verification_attempts[-1]

        last_checks = {
            check.command_id: check.status
            for check in last_attempt.checks
            if check.command_id
        }

        final_checks = {
            check.command_id: check.status
            for check in task.verification_checks
            if check.command_id
        }

        if not final_checks or last_checks != final_checks:
            outcome = {
                **outcome,
                "status": "insufficient_evidence",
                "reason": "final_checks_mismatch",
                "all_checks_passed": False,
            }

    return {
        "version": "v1.4.9-P4-D4-D",
        "run_id": record.run_id,
        "case_id": record.case_id,
        "base_commit_sha": record.base_commit_sha,
        "run_status": record.status,
        "task_id": task.task_id,
        "task_status": task.status,
        "verification_status": task.verification_status,
        "verification_repair_count": (task.verification_repair_count),
        "verification_repair_correction_count": (
            task.verification_repair_correction_count
        ),
        "outcome": outcome,
        # 历史 Run 没有保存模型请求的完整 Payload。
        # 不得仅根据代码实现推断历史模型输入已完整。
        "historical_input_coverage": "not_persisted",
        "historical_advisory_only": True,
        "task_completion_allowed": False,
        "experiment_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Repair Outcome Diagnostics")

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", default="TASK-005")
    parser.add_argument("--fe-agent-root", default=".")

    args = parser.parse_args()

    try:
        report = inspect_saved_repair(
            run_id=args.run_id,
            task_id=args.task_id,
            fe_agent_root=Path(args.fe_agent_root),
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Repair Diagnostics failed: {exc}") from exc

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
