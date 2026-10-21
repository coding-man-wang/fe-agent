from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> None:
    fe_agent_root = Path(__file__).resolve().parents[2]
    run_id = sys.argv[1] if len(sys.argv) > 1 else None

    runs_dir = fe_agent_root / ".fe-agent" / "benchmark" / "full_execution" / "runs"
    if run_id:
        run_path = runs_dir / f"{run_id}.json"
    else:
        candidates = sorted(
            runs_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError("没有 Full Execution Run Record")
        run_path = candidates[0]
        run_id = run_path.stem

    if not run_path.is_file():
        raise RuntimeError(f"Run Record 不存在: {run_path}")

    record = json.loads(run_path.read_text(encoding="utf-8"))
    print(f"run_id={run_id}")
    print(f"status={record.get('status')}")
    print(f"error={record.get('error')}")

    state_root = (
        fe_agent_root
        / ".fe-agent"
        / "benchmark"
        / "full_execution"
        / run_id
        / "state"
        / ".fe-agent"
        / "runtime"
    )

    for task in record.get("tasks", []):
        print("\\n=== Task ===")
        print(f"task_id={task.get('task_id')}")
        print(f"status={task.get('status')}")
        print(f"proposal_id={task.get('proposal_id')}")
        print(f"workspace_id={task.get('workspace_id')}")
        print(f"error={task.get('error')}")

        proposal_id = task.get("proposal_id")
        if not proposal_id:
            continue

        verification_path = state_root / "verifications" / f"{proposal_id}.json"
        if not verification_path.is_file():
            print("verification_record=<missing>")
            continue

        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        print(f"verification_status={verification.get('status')}")
        print(f"verification_error={verification.get('error')}")

        result = verification.get("result") or {}
        for check in result.get("checks", []):
            print("\\n--- Verification Check ---")
            print(f"name={check.get('name')}")
            print(f"status={check.get('status')}")
            print(check.get("details") or "<no details>")


if __name__ == "__main__":
    main()
