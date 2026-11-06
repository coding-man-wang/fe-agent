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
    "tests/test_v1_4_9_harness_bridge.py",
    "tests/test_v1_4_9_guarded_context_factory.py",
    "tests/test_v1_4_9_canary_policy.py",
]


REQUIRED_GATES = frozenset(
    {
        "rerank_diagnostic_complete",
        "oracle_symbols_resolved",
        "required_file_recall",
        "required_symbol_coverage",
        "no_file_recall_regression",
        "no_symbol_coverage_regression",
        "independent_symbol_evaluation",
        "noise_not_increased",
        "context_not_increased",
        "protected_retrieval_preserved",
        "shadow_selection_unchanged",
        "source_integrity",
        "metadata_integrity",
        "budget_integrity",
    }
)


def case_gates_passed(item: dict) -> bool:
    gates = item.get("completion_gates")

    return (
        item.get("status") == "passed"
        and isinstance(gates, dict)
        and REQUIRED_GATES.issubset(gates)
        and all(gates[name] is True for name in REQUIRED_GATES)
    )


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


def run_shadow_consumer_check() -> bool:
    """
    独立执行真实 Shadow + Consumer Benchmark。

    验证 P2-B 的候选预览确实通过，
    而不是只检查单元测试。
    """
    project_root = Path(__file__).resolve().parents[2]

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "apps.agent_cli." "benchmark_shadow_integration_v1_4_9",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    if process.returncode != 0:
        return False

    try:
        report = json.loads(process.stdout)
    except json.JSONDecodeError:
        return False

    cases = report.get("cases")

    if not isinstance(cases, list):
        return False

    expected_ids = {case.case_id for case in build_cases()}

    actual_ids = [item.get("case_id") for item in cases]

    return (
        report.get("status") == "passed"
        and len(actual_ids) == len(expected_ids)
        and set(actual_ids) == expected_ids
        and all(
            item.get("passed") is True
            and item.get("context_policy_valid") is True
            and item.get("snapshot_gate", {}).get("allowed") is True
            for item in cases
        )
    )


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

    expected_ids = {case.case_id for case in build_cases()}

    actual_ids = [item.get("case_id") for item in reports]

    all_cases_passed = (
        len(actual_ids) == len(expected_ids)
        and set(actual_ids) == expected_ids
        and all(case_gates_passed(item) for item in reports)
    )

    # 只有 P3 本身通过后，
    # 才额外运行真实 Shadow Consumer。
    shadow_consumer_passed = (
        run_shadow_consumer_check() if safety_passed and all_cases_passed else False
    )

    approved = (
        safety_passed
        and all_cases_passed
        and revision == revision_after
        and not dirty_after
        and shadow_consumer_passed
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
        "shadow_consumer_passed": shadow_consumer_passed,
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
