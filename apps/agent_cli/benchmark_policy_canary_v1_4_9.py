from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent_core.benchmark.context_p3_cases import build_cases
from apps.agent_cli.benchmark_multi_task_v1_4_8 import (
    git_command,
    make_services,
    run_case,
)
from apps.agent_cli.benchmark_symbol_window_v1_4_6 import (
    resolve_repository_root,
)

SAFETY_TESTS = [
    "tests/test_v1_4_8_graph_evidence_filter.py",
    "tests/test_v1_4_8_p3_multi_task.py",
    "tests/test_v1_4_9_selection_shadow_integration.py",
    "tests/test_v1_4_9_selection_snapshot_gate.py",
    "tests/test_v1_4_9_context_policy_runtime.py",
]


def run_safety_tests() -> bool:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *SAFETY_TESTS,
            "-q",
        ],
        check=False,
    )
    return process.returncode == 0


async def main():
    load_dotenv()

    root = resolve_repository_root()
    revision = git_command(root, "rev-parse", "HEAD")

    # 不允许在被测试仓库上有未提交的受跟踪文件修改。
    dirty = git_command(
        root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )

    if dirty:
        raise RuntimeError("Canary 仓库存在未提交的受跟踪文件修改")

    safety_passed = run_safety_tests()

    reports = []

    if safety_passed:
        selection_service, rerank_runner = make_services()

        for case in build_cases():
            try:
                result = await run_case(
                    case=case,
                    root=root,
                    revision=revision,
                    selection_service=selection_service,
                    rerank_runner=rerank_runner,
                )
            except Exception as exc:
                result = {
                    "case_id": case.case_id,
                    "status": "error",
                    "reason": type(exc).__name__,
                }

            reports.append(result)

    revision_after = git_command(root, "rev-parse", "HEAD")

    dirty_after = git_command(
        root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )

    all_cases_passed = len(reports) == len(build_cases()) and all(
        item.get("status") == "passed"
        and all(item.get("completion_gates", {}).values())
        for item in reports
    )

    approved = (
        safety_passed
        and all_cases_passed
        and revision == revision_after
        and not dirty_after
    )

    output = {
        "version": "v1.4.9-P2-C",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repository_revision": revision,
        "status": ("passed" if approved else "failed"),
        "safety_tests_passed": safety_passed,
        "total_cases": len(build_cases()),
        "passed_cases": sum(item.get("status") == "passed" for item in reports),
        "cases": [
            {
                "case_id": item["case_id"],
                "status": item["status"],
                "completion_gates": item.get("completion_gates", {}),
            }
            for item in reports
        ],
        "active_experiment_eligible": approved,
        # 不等于生产环境已授权。
        "production_ready": False,
    }

    destination = Path("var/canary/context_policy_v1_4_9.json")
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
    )

    if not approved:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
