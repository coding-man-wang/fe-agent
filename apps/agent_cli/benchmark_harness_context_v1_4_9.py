from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.benchmark.context_p3_cases import build_cases
from agent_core.benchmark.context_policy_runtime import (
    ContextRolloutConfig,
    GuardedContextPolicyConsumer,
)
from agent_core.benchmark.harness_context_policy_bridge import (
    HarnessContextPolicyBridge,
)
from agent_core.benchmark.selection_shadow_integration import (
    ContextOptimizationShadowRunner,
    ShadowPolicyConfig,
    ShadowTaskContextSelectionService,
)
from agent_core.benchmark.shadow_audit import JsonlShadowAuditStore
from agent_core.change_surface.context_builder import (
    ResolvedTaskContextBuilder,
)
from agent_core.change_surface.investigation import (
    InvestigationArtifactBuilder,
)
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.change_surface.resolver import ChangeSurfaceResolver
from agent_core.code_intelligence.repository_snapshot import (
    RepositorySymbolSnapshotBuilder,
)
from agent_core.codebase.model import RepositoryProfile
from agent_core.context.builder import ProjectContextBuilder
from agent_core.harness.coordinator import HarnessV1Coordinator
from agent_core.harness.trace import JsonlHarnessTraceStore
from agent_core.coding.patch_model import PatchPlan, PatchTarget

from apps.agent_cli.benchmark_active_context_v1_4_9 import (
    evaluate_bundle,
    quality_gate,
    validate_canary_report,
)
from apps.agent_cli.benchmark_multi_task_v1_4_8 import (
    git_command,
    make_services,
)
from apps.agent_cli.benchmark_symbol_window_v1_4_6 import (
    resolve_repository_root,
)


class ForbiddenLLM:
    async def chat(self, request):
        raise AssertionError("P4-A 使用固定 Surface Fixture，" "不应调用 LLM Resolver")


def surface_for(case_id: str) -> ResolvedChangeSurface:
    """
    固定的工程场景 Fixture。
    不读取 Oracle，不参与文件检索和过滤评分。
    """

    if case_id == "ORDER-001":
        return ResolvedChangeSurface(
            task_id=case_id,
            implementation_targets=[
                "src/views/OrdersView.vue",
            ],
            dependencies=[
                "src/api/orders.ts",
                "src/mocks/service.ts",
                "src/types/domain.ts",
            ],
            references=[
                "src/api/requestClient.ts",
            ],
        )

    if case_id == "USER-001":
        return ResolvedChangeSurface(
            task_id=case_id,
            implementation_targets=[
                "src/views/UsersView.vue",
                "src/components/UserFilters.vue",
                "src/stores/users.ts",
            ],
            dependencies=[
                "src/api/users.ts",
                "src/mocks/service.ts",
                "src/types/domain.ts",
            ],
            references=[
                "src/components/PaginationBar.vue",
            ],
            tests=[
                "tests/unit/usersStore.test.ts",
            ],
        )

    if case_id == "API-001":
        return ResolvedChangeSurface(
            task_id=case_id,
            implementation_targets=[
                "src/api/requestClient.ts",
            ],
            dependencies=[
                "src/api/orders.ts",
                "src/api/users.ts",
                "src/types/domain.ts",
            ],
            references=[
                "AGENTS.md",
            ],
        )

    raise ValueError(f"未知 Case: {case_id}")


def verify_scope(
    *,
    bridge: HarnessContextPolicyBridge,
    consumption,
    surface,
    root: Path,
    task_id: str,
) -> bool:
    """
    使用真实 Patch Scope Guard。
    只验证权限，不应用 Patch。
    """
    allowed_path = "src/views/OrdersView.vue"
    forbidden_path = "src/components/UserTable.vue"

    def plan_for(path: str) -> PatchPlan:
        digest = hashlib.sha256((root / path).read_bytes()).hexdigest()

        return PatchPlan(
            task_id=task_id,
            artifact_id="p4-scope-probe",
            context_hash="p4-context-probe",
            targets=[
                PatchTarget(
                    relative_path=path,
                    action="modify",
                    expected_sha256=digest,
                    objective="只验证修改范围",
                    proposed_approach="不执行任何写入",
                )
            ],
        )

    allow = bridge.guard_patch_plan(
        context=consumption,
        plan=plan_for(allowed_path),
        surface=surface,
    )

    deny = bridge.guard_patch_plan(
        context=consumption,
        plan=plan_for(forbidden_path),
        surface=surface,
    )

    return allow.allowed is True and deny.allowed is False


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-revision", required=True)
    args = parser.parse_args()

    load_dotenv()

    project_root = Path(__file__).resolve().parents[2]
    root = resolve_repository_root()
    revision = git_command(root, "rev-parse", "HEAD")

    if revision != args.approved_revision:
        raise SystemExit("目标 Repository Revision 不匹配")

    if git_command(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("目标仓库存在未提交的受跟踪文件修改")

    cases = build_cases()
    canary_path = project_root / "var" / "canary" / "context_policy_v1_4_9.json"

    if not canary_path.exists():
        raise SystemExit("缺少 P2-C Canary 报告")

    canary = json.loads(canary_path.read_text(encoding="utf-8"))

    if not validate_canary_report(
        canary,
        revision=revision,
        expected_ids={c.case_id for c in cases},
    ):
        raise SystemExit("Canary 无效，禁止 Active 实验")

    run_dir = project_root / "var" / "p4" / uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)

    audit = JsonlShadowAuditStore(path=run_dir / "audit.jsonl")
    trace = JsonlHarnessTraceStore(path=run_dir / "trace.jsonl")

    # 使用真实 HarnessV1Coordinator。
    # P4-A 的 Surface 是固定 Fixture，
    # 所以模型解析器被显式禁止调用。
    forbidden_llm = ForbiddenLLM()

    coordinator = HarnessV1Coordinator(
        store=JsonTaskArtifactStore(root=run_dir / "artifacts"),
        investigation_builder=InvestigationArtifactBuilder(llm=forbidden_llm),
        surface_resolver=ChangeSurfaceResolver(llm=forbidden_llm),
        trace_store=trace,
    )

    selection_service, rerank_runner = make_services()

    repository = RepositoryProfile(
        project_name="fe-agent-benchmark-web",
        root_path=str(root),
        framework="Vue3",
        language="TypeScript",
        build_tool="Vite",
        package_manager="npm",
    )

    results = []

    for case in cases:
        reports = []

        def emit(report):
            audit.append(report)
            reports.append(report)

        wrapper = ShadowTaskContextSelectionService(
            delegate=selection_service,
            config=ShadowPolicyConfig(
                mode="shadow",
                timeout_seconds=90,
            ),
            runner=ContextOptimizationShadowRunner(rerank_runner=rerank_runner),
            emit=emit,
        )

        try:
            selection_result = await wrapper.select(
                repository_root=root,
                repository_revision=revision,
                task=case.task,
                raw_requirement=case.raw_requirement,
                requirement_spec=case.requirement,
            )

            if len(reports) != 1:
                raise RuntimeError("missing_shadow_report")

            report = reports[0]
            surface = surface_for(case.case_id)

            plain_bundle = ProjectContextBuilder().build(
                repository_root=root,
                requirement=case.requirement,
                technical_spec=case.technical_spec,
                task=case.task,
                repository=repository,
                selection=selection_result.selection,
                max_files=8,
                max_total_chars=40_000,
                max_chars_per_file=10_000,
            )

            resolved_builder = ResolvedTaskContextBuilder(
                artifact_resolver=coordinator.artifact_resolver,
                context_policy=coordinator.context_policy,
            )

            # Harness 的原始 Resolved Context，
            # 与 Active 采用结果进行同类型比较。
            baseline = resolved_builder.build(
                repository_root=root,
                requirement=case.requirement,
                technical_spec=case.technical_spec,
                task=case.task,
                repository=repository,
                index=selection_result.index,
                selection=selection_result.selection,
                surface=surface,
            )

            is_active = case.case_id == "ORDER-001"

            config = (
                ContextRolloutConfig(
                    mode="active",
                    active_enabled=True,
                    canary_passed=True,
                    approved_revision=revision,
                    task_allowlist=frozenset({"ORDER-001"}),
                )
                if is_active
                else ContextRolloutConfig(mode="shadow")
            )

            bridge = HarnessContextPolicyBridge(
                coordinator=coordinator,
                consumer=GuardedContextPolicyConsumer(
                    config=config,
                    audit_sink=audit.append,
                ),
            )

            surface_before = surface.model_dump(mode="json")

            consumption = bridge.build_resolved_context(
                repository_root=root,
                requirement=case.requirement,
                technical_spec=case.technical_spec,
                repository=repository,
                index=selection_result.index,
                original_bundle=plain_bundle,
                selection_result=selection_result,
                surface=surface,
                raw_requirement=case.raw_requirement,
                report=report,
                get_current_revision=lambda: git_command(root, "rev-parse", "HEAD"),
                fresh_in_process_report=True,
            )

            snapshot = RepositorySymbolSnapshotBuilder().build(
                file_index=selection_result.index,
                repository_revision=revision,
            )

            baseline_quality = evaluate_bundle(
                case=case,
                bundle=baseline,
                repository_root=root,
                symbol_index=snapshot.symbol_index,
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

            quality_ok = quality_gate(
                baseline_quality,
                actual_quality,
            )

            if is_active:
                policy_ok = (
                    consumption.decision.status == "active"
                    and actual_quality["context_chars"]
                    < baseline_quality["context_chars"]
                    and bool(actual_drops)
                    and set(actual_drops).issubset(
                        set(report["effective_proposed_drops"])
                    )
                )
            else:
                policy_ok = (
                    consumption.decision.status == "shadow"
                    and baseline_quality["files"] == actual_quality["files"]
                    and baseline_quality["context_chars"]
                    == actual_quality["context_chars"]
                )

            scope_ok = (
                verify_scope(
                    bridge=bridge,
                    consumption=consumption,
                    surface=surface,
                    root=root,
                    task_id=case.task.task_id,
                )
                if is_active
                else True
            )

            surface_unchanged = (
                surface.model_dump(mode="json")
                == surface_before
                == consumption.bundle.change_surface
            )

            passed = (
                quality_ok
                and policy_ok
                and scope_ok
                and surface_unchanged
                and consumption.decision.snapshot_gate_reason == "snapshot_valid"
            )

            results.append(
                {
                    "case_id": case.case_id,
                    "status": "passed" if passed else "failed",
                    "policy_status": consumption.decision.status,
                    "policy_reason": consumption.decision.reason,
                    "baseline": baseline_quality,
                    "actual": actual_quality,
                    "actual_drops": actual_drops,
                    "quality_ok": quality_ok,
                    "policy_ok": policy_ok,
                    "scope_ok": scope_ok,
                    "surface_unchanged": surface_unchanged,
                }
            )

        except Exception as exc:
            results.append(
                {
                    "case_id": case.case_id,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    context_events = [
        event for event in trace.read_all() if event.event_type == "context_refined"
    ]

    trace_ok = len(context_events) == len(cases) and all(
        any(
            event.task_id == case["case_id"]
            and event.data["context_files"] == case["actual"]["files"]
            and event.data["policy_status"] == case["policy_status"]
            for event in context_events
        )
        for case in results
        if case["status"] == "passed"
    )

    unchanged_repository = git_command(
        root, "rev-parse", "HEAD"
    ) == revision and not git_command(
        root, "status", "--porcelain", "--untracked-files=no"
    )

    passed = (
        len(results) == len(cases)
        and all(item["status"] == "passed" for item in results)
        and trace_ok
        and unchanged_repository
    )

    output = {
        "version": "v1.4.9-P4-A",
        "status": "passed" if passed else "failed",
        "production_ready": False,
        "repository_revision": revision,
        "trace_ok": trace_ok,
        "repository_unchanged": unchanged_repository,
        "cases": results,
    }

    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
