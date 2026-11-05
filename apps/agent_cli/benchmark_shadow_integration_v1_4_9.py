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
                ),
            }
        )

    output = {
        "version": "v1.4.9-P1",
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
