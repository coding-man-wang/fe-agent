from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from agent_core.benchmark.case_loader import (
    BenchmarkCaseLoader,
)
from agent_core.benchmark.execution_model import (
    FullExecutionRunRecord,
)
from agent_core.benchmark.test_scope_alignment import (
    analyze_test_scope_alignment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline Test Scope Alignment Diagnostics"
    )

    parser.add_argument("--case-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fe-agent-root", default=".")

    args = parser.parse_args()

    if re.fullmatch(r"CASE-[0-9]{3}", args.case_id) is None:
        parser.error("Invalid Case ID")

    if re.fullmatch(r"[0-9a-f]{32}", args.run_id) is None:
        parser.error("Invalid Run ID")

    root = Path(args.fe_agent_root).resolve()

    case = BenchmarkCaseLoader(root / "benchmark").load(args.case_id)

    path = (
        root
        / ".fe-agent"
        / "benchmark"
        / "full_execution"
        / "runs"
        / f"{args.run_id}.json"
    )

    try:
        run = FullExecutionRunRecord.model_validate_json(
            path.read_text(encoding="utf-8-sig")
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot read execution record: {exc}") from exc

    if run.run_id != args.run_id:
        raise SystemExit("Run ID mismatch")

    report = analyze_test_scope_alignment(
        case=case,
        run=run,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
