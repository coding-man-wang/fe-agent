from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Any

from agent_core.benchmark.context_policy_runtime import (
    ContextConsumptionResult,
    GuardedContextPolicyConsumer,
)
from agent_core.change_surface.context_builder import (
    ResolvedTaskContextBuilder,
)
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.codebase.model import (
    RepositoryFileIndex,
    RepositoryProfile,
)
from agent_core.codebase.task_context_selection import (
    TaskContextSelectionResult,
)
from agent_core.context.model import TaskContextBundle
from agent_core.harness.coordinator import HarnessV1Coordinator
from agent_core.harness.trace import HarnessTraceEvent
from agent_core.sdd.model import (
    RequirementSpec,
    TaskPlan,
    TechnicalSpec,
)


class HarnessContextPolicyBridge:
    """
    v1.4.9-P4

    在现有 ResolvedTaskContextBuilder 外包一层
    Context Policy Consumer。

    不修改 Change Surface。
    不替代 Scope Guard。
    不绕过 TaskCompletionGate。
    """

    def __init__(
        self,
        *,
        coordinator: HarnessV1Coordinator,
        consumer: GuardedContextPolicyConsumer,
    ):
        self.coordinator = coordinator
        self.consumer = consumer

    def build_resolved_context(
        self,
        *,
        repository_root: str | Path,
        requirement: RequirementSpec,
        technical_spec: TechnicalSpec,
        repository: RepositoryProfile,
        index: RepositoryFileIndex,
        original_bundle: TaskContextBundle,
        selection_result: TaskContextSelectionResult,
        surface: ResolvedChangeSurface,
        raw_requirement: str,
        report: Mapping[str, Any],
        get_current_revision: Callable[[], str],
        engineering_rules: list[str] | None = None,
        task_plan: TaskPlan | None = None,
        fresh_in_process_report: bool = False,
    ) -> ContextConsumptionResult:

        task = original_bundle.task

        if surface.task_id != task.task_id:
            raise ValueError("Change Surface 和 Task ID 不一致")

        # 复用 Harness 已有的 ArtifactResolver
        # 和 RoleAwareContextPolicy。
        builder = ResolvedTaskContextBuilder(
            artifact_resolver=self.coordinator.artifact_resolver,
            context_policy=self.coordinator.context_policy,
        )

        indexable = {
            record.relative_path for record in index.files if record.is_indexable
        }

        # 已存在且可索引的变更目标、测试文件
        # 必须在实际 Context 中保留。
        mandatory = (
            set(surface.implementation_targets) | set(surface.tests)
        ) & indexable

        # Surface 可能补入原 Retrieval 未包含的文件。
        # P2-B 需要额外校验这些文件的内容变化。
        surface_files = (
            set(surface.implementation_targets)
            | set(surface.dependencies)
            | set(surface.references)
            | set(surface.tests)
        ) & indexable

        def build_context(selection):
            return builder.build(
                repository_root=repository_root,
                requirement=requirement,
                technical_spec=technical_spec,
                task=task,
                repository=repository,
                index=index,
                selection=selection,
                surface=surface,
                engineering_rules=engineering_rules,
                task_plan=task_plan,
            )

        result = self.consumer.consume(
            repository_root=repository_root,
            task=task,
            raw_requirement=raw_requirement,
            selection_result=selection_result,
            report=report,
            get_current_revision=get_current_revision,
            build_context=build_context,
            required_context_paths=mandatory,
            additional_source_paths=surface_files,
            fresh_in_process_report=fresh_in_process_report,
        )

        # 只记录真正返回给下游的 Context。
        # 避免 Shadow 的 Candidate Preview
        # 被记录为已经实际使用的 Context。
        if self.coordinator.trace_store is not None:
            self.coordinator.trace_store.append(
                HarnessTraceEvent(
                    task_id=task.task_id,
                    event_type="context_refined",
                    data={
                        "context_files": [
                            item.relative_path for item in result.bundle.code_context
                        ],
                        "policy_mode": (result.decision.mode),
                        "policy_status": (result.decision.status),
                        "policy_reason": (result.decision.reason),
                    },
                )
            )

        return result

    def guard_patch_plan(
        self,
        *,
        context: ContextConsumptionResult,
        plan,
        surface: ResolvedChangeSurface,
    ):
        # Context 没通过构建门禁时，
        # 不允许继续使用它生成 Patch。
        if (
            context.bundle.task.task_id != plan.task_id
            or surface.task_id != plan.task_id
        ):
            raise ValueError("Context、Patch Plan、Surface 的 Task ID 不一致")

        # 真正的修改范围校验仍由已有 Harness 负责。
        return self.coordinator.guard_patch_plan(
            plan=plan,
            surface=surface,
        )

    @staticmethod
    def complete_task(
        *,
        context: ContextConsumptionResult,
        completion_gate,
        proposal,
        journal,
        verification,
    ):
        if context.bundle.task.task_id != proposal.task_id:
            raise ValueError("Context 与 Completion Proposal 的 Task ID 不一致")

        # 不能把上下文策略校验当成测试或审批。
        # 仍要求原有 Approval / Patch / Verification /
        # Repository Final State 全部通过。
        return completion_gate.complete(
            proposal=proposal,
            journal=journal,
            verification=verification,
        )
