from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from agent_core.benchmark.context_p3_cases import (
    build_cases,
)
from agent_core.benchmark.selection_shadow_integration import (
    ContextOptimizationShadowRunner,
    ShadowPolicyConfig,
    ShadowTaskContextSelectionService,
)
from apps.agent_cli.benchmark_multi_task_v1_4_8 import (
    git_command,
    make_services,
)
from apps.agent_cli.benchmark_symbol_window_v1_4_6 import (
    resolve_repository_root,
)
import os

from agent_core.benchmark.shadow_audit import (
    JsonlShadowAuditStore,
)
from agent_core.benchmark.selection_snapshot_gate import (
    ShadowSnapshotGate,
)
from agent_core.benchmark.context_policy_runtime import (
    ContextRolloutConfig,
    GuardedContextPolicyConsumer,
)
from agent_core.context.builder import ProjectContextBuilder
from agent_core.codebase.model import RepositoryProfile


async def main():
    load_dotenv()

    root = resolve_repository_root()
    revision = git_command(root, "rev-parse", "HEAD")

    # 复用 P3 已验证的 Graph/Symbol 配置。
    delegate, rerank_runner = make_services()

    reports = []

    audit_path = os.getenv(
        "FE_AGENT_SHADOW_AUDIT_PATH",
        "var/audit/context_shadow_v1_4_9.jsonl",
    )

    audit_store = JsonlShadowAuditStore(
        path=audit_path,
    )

    audited_task_ids: set[str] = set()

    def emit_report(report: dict):
        reports.append(report)

        # 只有持久化成功才记录完成。
        audit_store.append(report)

        audited_task_ids.add(report["task_id"])

    service = ShadowTaskContextSelectionService(
        delegate=delegate,
        config=ShadowPolicyConfig(
            mode="shadow",
            timeout_seconds=90.0,
        ),
        runner=ContextOptimizationShadowRunner(
            rerank_runner=rerank_runner,
        ),
        emit=emit_report,
    )

    policy_consumer = GuardedContextPolicyConsumer(
        config=ContextRolloutConfig(
            mode="shadow",  # 本轮默认保持 shadow
            active_enabled=False,
            canary_passed=False,
        ),
        audit_sink=audit_store.append,
    )

    context_builder = ProjectContextBuilder()

    repository = RepositoryProfile(
        project_name="fe-agent-benchmark-web",
        root_path=str(root),
        framework="Vue3",
        language="TypeScript",
        build_tool="Vite",
        package_manager="npm",
    )

    snapshot_gate = ShadowSnapshotGate()

    results = []

    for case in build_cases():
        before = len(reports)

        result = await service.select(
            repository_root=root,
            repository_revision=revision,
            task=case.task,
            raw_requirement=case.raw_requirement,
            requirement_spec=case.requirement,
        )

        report = reports[-1] if len(reports) > before else {"status": "missing_report"}

        snapshot_decision = snapshot_gate.validate(
            report=report,
            repository_root=root,
            selection=result.selection,
            task_id=case.case_id,
            raw_requirement=case.raw_requirement,
            expected_revision=revision,
            current_revision=git_command(root, "rev-parse", "HEAD"),
        )

        def build_context(selection):
            return context_builder.build(
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

        consumption = policy_consumer.consume(
            repository_root=root,
            task=case.task,
            raw_requirement=case.raw_requirement,
            selection_result=result,
            report=report,
            get_current_revision=lambda: git_command(root, "rev-parse", "HEAD"),
            build_context=build_context,
            fresh_in_process_report=True,
        )

        binding = report.get("evidence_binding") or {}

        evidence_valid = (
            binding.get("source_matched") is True
            and binding.get("selection_matched") is True
        )

        audit_written = case.case_id in audited_task_ids

        results.append(
            {
                "case_id": case.case_id,
                "original_selection": [
                    item.relative_path for item in result.selection.selected_files
                ],
                "shadow_report": report,
                "evidence_valid": evidence_valid,
                "audit_written": audit_written,
                "passed": (
                    report.get("status") == "complete"
                    and report.get("applied") is False
                    and report.get("selection_unchanged") is True
                    and evidence_valid
                    and audit_written
                    and snapshot_decision.allowed
                ),
                "snapshot_gate": {
                    "allowed": snapshot_decision.allowed,
                    "reason": snapshot_decision.reason,
                },
                "context_policy": {
                    "status": consumption.decision.status,
                    "reason": consumption.decision.reason,
                    "files_before": list(consumption.decision.context_files_before),
                    "files_after": list(consumption.decision.context_files_after),
                },
            }
        )

    output = {
        "version": "v1.4.9-P2-A",
        "mode": "shadow",
        "repository_revision": revision,
        "total_cases": len(results),
        "passed_cases": sum(item["passed"] for item in results),
        "status": ("passed" if all(item["passed"] for item in results) else "failed"),
        "cases": results,
    }

    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
    )

    if output["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
