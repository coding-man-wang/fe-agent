from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent_core.benchmark.context_file_quality import (
    ContextFileQualityEvaluator,
)
from agent_core.benchmark.context_p3_cases import build_cases
from agent_core.benchmark.context_policy_runtime import (
    ContextRolloutConfig,
    GuardedContextPolicyConsumer,
)
from agent_core.benchmark.selection_evidence_binding import (
    capture_selection_evidence,
)
from agent_core.benchmark.selection_shadow_integration import (
    ContextOptimizationShadowRunner,
    ShadowPolicyConfig,
    ShadowTaskContextSelectionService,
)
from agent_core.benchmark.shadow_audit import (
    JsonlShadowAuditStore,
)
from agent_core.benchmark.symbol_window_evaluator import (
    SymbolWindowBenchmarkCase,
    SymbolWindowBenchmarkEvaluator,
)
from agent_core.code_intelligence.repository_snapshot import (
    RepositorySymbolSnapshotBuilder,
)
from agent_core.codebase.model import RepositoryProfile
from agent_core.context.builder import ProjectContextBuilder

from apps.agent_cli.benchmark_multi_task_v1_4_8 import (
    git_command,
    make_services,
)
from apps.agent_cli.benchmark_symbol_window_v1_4_6 import (
    resolve_repository_root,
)

ACTIVE_CASE_ID = "ORDER-001"

REQUIRED_CANARY_GATES = frozenset(
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


def validate_canary_report(
    report: dict,
    *,
    revision: str,
    expected_ids: set[str],
) -> bool:
    """只接受匹配当前目标 Revision 的完整 Canary 报告。"""

    cases = report.get("cases")

    if not isinstance(cases, list):
        return False

    actual_ids = [item.get("case_id") for item in cases if isinstance(item, dict)]

    if (
        report.get("status") != "passed"
        or report.get("repository_revision") != revision
        or report.get("safety_tests_passed") is not True
        or report.get("shadow_consumer_passed") is not True
        or report.get("active_experiment_eligible") is not True
        or report.get("production_ready") is not False
        or report.get("total_cases") != len(expected_ids)
        or report.get("passed_cases") != len(expected_ids)
        or len(actual_ids) != len(expected_ids)
        or set(actual_ids) != expected_ids
    ):
        return False

    for item in cases:
        gates = item.get("completion_gates")

        if (
            item.get("status") != "passed"
            or not isinstance(gates, dict)
            or not REQUIRED_CANARY_GATES.issubset(gates)
            or not all(gates[name] is True for name in REQUIRED_CANARY_GATES)
        ):
            return False

    return True


def evaluate_bundle(
    *,
    case,
    bundle,
    repository_root: Path,
    symbol_index,
) -> dict:
    """Oracle 只在这里参与验收，不参与策略决策。"""

    file_quality = ContextFileQualityEvaluator().evaluate(
        bundle=bundle,
        oracle=case.oracle,
    )

    symbol_quality = SymbolWindowBenchmarkEvaluator().evaluate(
        case=SymbolWindowBenchmarkCase(
            case_id=f"{case.case_id}-ACTIVE-E2E",
            required_symbol_ids=list(case.required_symbol_ids),
            require_symbol_window=False,
            require_independent_symbol_index=True,
        ),
        bundle=bundle,
        repository_root=repository_root,
        symbol_index=symbol_index,
    )

    return {
        "files": [item.relative_path for item in bundle.code_context],
        "context_chars": bundle.budget.used_chars,
        "required_file_recall": file_quality.recall,
        "missing_required_files": file_quality.missing_required,
        "irrelevant_chars": file_quality.irrelevant_chars,
        "required_symbol_coverage": symbol_quality.symbol_coverage,
        "missing_required_symbols": symbol_quality.missing_symbol_ids,
        "symbol_status": symbol_quality.status,
        "source_integrity": (symbol_quality.source_integrity_preserved),
        "metadata_integrity": (symbol_quality.metadata_integrity_preserved),
        "budget_integrity": symbol_quality.budget_preserved,
    }


def quality_gate(baseline: dict, actual: dict) -> bool:
    """必须同时满足绝对覆盖率和无回归要求。"""

    return (
        baseline["required_file_recall"] == 1.0
        and baseline["required_symbol_coverage"] == 1.0
        and actual["required_file_recall"] == 1.0
        and actual["required_symbol_coverage"] == 1.0
        and actual["symbol_status"] == "passed"
        and actual["source_integrity"] is True
        and actual["metadata_integrity"] is True
        and actual["budget_integrity"] is True
        and actual["irrelevant_chars"] <= baseline["irrelevant_chars"]
        and actual["context_chars"] <= baseline["context_chars"]
    )


async def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--active-order",
        action="store_true",
        help="仅允许 ORDER-001 执行 Active 实验",
    )
    parser.add_argument(
        "--approved-revision",
        required=True,
        help="明确批准的 benchmark-web Git Revision",
    )

    args = parser.parse_args()

    if not args.active_order:
        raise SystemExit("必须显式传入 --active-order 才能运行该实验")

    load_dotenv()

    project_root = Path(__file__).resolve().parents[2]
    root = resolve_repository_root()
    revision = git_command(root, "rev-parse", "HEAD")

    if args.approved_revision != revision:
        raise SystemExit("批准的 Revision 与当前目标仓库不一致")

    if git_command(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("目标仓库存在未提交的受跟踪文件修改")

    cases = build_cases()
    expected_ids = {case.case_id for case in cases}

    canary_path = project_root / "var" / "canary" / "context_policy_v1_4_9.json"

    if not canary_path.is_file():
        raise SystemExit("缺少 Canary 报告，请先执行 P2-C Benchmark")

    canary = json.loads(canary_path.read_text(encoding="utf-8"))

    if not validate_canary_report(
        canary,
        revision=revision,
        expected_ids=expected_ids,
    ):
        raise SystemExit("Canary 不完整、未通过或目标 Revision 已变化")

    selection_service, rerank_runner = make_services()

    repository = RepositoryProfile(
        project_name="fe-agent-benchmark-web",
        root_path=str(root),
        framework="Vue3",
        language="TypeScript",
        build_tool="Vite",
        package_manager="npm",
    )

    builder = ProjectContextBuilder()

    audit_store = JsonlShadowAuditStore(
        path=(project_root / "var" / "audit" / "active_context_v1_4_9.jsonl"),
    )

    results = []

    for case in cases:
        reports: list[dict] = []
        audited_ids: set[str] = set()

        def emit_report(report: dict) -> None:
            reports.append(report)
            audit_store.append(report)
            audited_ids.add(report["task_id"])

        wrapper = ShadowTaskContextSelectionService(
            delegate=selection_service,
            config=ShadowPolicyConfig(
                mode="shadow",
                timeout_seconds=90.0,
            ),
            runner=ContextOptimizationShadowRunner(
                rerank_runner=rerank_runner,
            ),
            emit=emit_report,
        )

        try:
            selected = await wrapper.select(
                repository_root=root,
                repository_revision=revision,
                task=case.task,
                raw_requirement=case.raw_requirement,
                requirement_spec=case.requirement,
            )

            if len(reports) != 1:
                raise RuntimeError("missing_shadow_report")

            shadow_report = reports[0]

            original_dump = selected.selection.model_dump(mode="json")

            before_fingerprint = capture_selection_evidence(
                repository_root=root,
                selection=selected.selection,
            )

            # 复用真实 Symbol Index，
            # 但所有 Oracle 评估独立于过滤算法执行。
            snapshot = RepositorySymbolSnapshotBuilder().build(
                file_index=selected.index,
                repository_revision=revision,
            )

            def build_context(selection):
                return builder.build(
                    repository_root=root,
                    requirement=case.requirement,
                    technical_spec=case.technical_spec,
                    task=case.task,
                    repository=repository,
                    selection=selection,
                    engineering_rules=[],
                    max_files=8,
                    max_total_chars=40_000,
                    max_chars_per_file=10_000,
                )

            baseline_bundle = build_context(selected.selection.model_copy(deep=True))

            baseline_quality = evaluate_bundle(
                case=case,
                bundle=baseline_bundle,
                repository_root=root,
                symbol_index=snapshot.symbol_index,
            )

            # 只对 ORDER 使用 Active。
            is_active_case = case.case_id == ACTIVE_CASE_ID

            if is_active_case:
                config = ContextRolloutConfig(
                    mode="active",
                    active_enabled=True,
                    canary_passed=True,
                    approved_revision=revision,
                    task_allowlist=frozenset({ACTIVE_CASE_ID}),
                    max_drops=2,
                )
            else:
                config = ContextRolloutConfig(
                    mode="shadow",
                    active_enabled=False,
                )

            consumer = GuardedContextPolicyConsumer(
                config=config,
                audit_sink=audit_store.append,
            )

            consumption = consumer.consume(
                repository_root=root,
                task=case.task,
                raw_requirement=case.raw_requirement,
                selection_result=selected,
                report=shadow_report,
                get_current_revision=lambda: git_command(root, "rev-parse", "HEAD"),
                build_context=build_context,
                fresh_in_process_report=(case.task.task_id in audited_ids),
            )

            actual_quality = evaluate_bundle(
                case=case,
                bundle=consumption.bundle,
                repository_root=root,
                symbol_index=snapshot.symbol_index,
            )

            before_paths = set(baseline_quality["files"])
            after_paths = set(actual_quality["files"])

            actual_drops = sorted(before_paths - after_paths)

            after_fingerprint = capture_selection_evidence(
                repository_root=root,
                selection=selected.selection,
            )

            unchanged = (
                original_dump == selected.selection.model_dump(mode="json")
                and before_fingerprint == after_fingerprint
                and git_command(root, "rev-parse", "HEAD") == revision
            )

            quality_ok = quality_gate(
                baseline_quality,
                actual_quality,
            )

            if is_active_case:
                policy_ok = (
                    consumption.decision.status == "active"
                    and len(actual_drops) > 0
                    and set(actual_drops)
                    == set(shadow_report.get("effective_proposed_drops", []))
                )
            else:
                policy_ok = (
                    consumption.decision.status == "shadow"
                    and before_paths == after_paths
                    and baseline_quality["context_chars"]
                    == actual_quality["context_chars"]
                )

            passed = (
                shadow_report.get("status") == "complete"
                and shadow_report.get("selection_unchanged") is True
                and quality_ok
                and policy_ok
                and unchanged
            )

            results.append(
                {
                    "case_id": case.case_id,
                    "status": "passed" if passed else "failed",
                    "policy_status": consumption.decision.status,
                    "policy_reason": consumption.decision.reason,
                    "snapshot_gate_reason": (consumption.decision.snapshot_gate_reason),
                    "actual_drops": actual_drops,
                    "quality_gate_passed": quality_ok,
                    "policy_gate_passed": policy_ok,
                    "source_unchanged": unchanged,
                    "baseline": baseline_quality,
                    "actual": actual_quality,
                }
            )

        except Exception as exc:
            results.append(
                {
                    "case_id": case.case_id,
                    "status": "error",
                    "error_type": type(exc).__name__,
                }
            )

    final_revision = git_command(root, "rev-parse", "HEAD")

    clean = not git_command(root, "status", "--porcelain", "--untracked-files=no")

    approved = (
        len(results) == len(cases)
        and all(item["status"] == "passed" for item in results)
        and final_revision == revision
        and clean
    )

    output = {
        "version": "v1.4.9-P3-E2E",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repository_revision": revision,
        "mode": "controlled_active_experiment",
        "active_task": ACTIVE_CASE_ID,
        "total_cases": len(results),
        "passed_cases": sum(item["status"] == "passed" for item in results),
        "status": "passed" if approved else "failed",
        "production_ready": False,
        "cases": results,
    }

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
