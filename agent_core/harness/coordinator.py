from pathlib import Path

from agent_core.artifacts.resolver import TaskArtifactResolver
from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.change_surface.context_builder import ResolvedTaskContextBuilder
from agent_core.change_surface.context_policy import RoleAwareContextPolicy
from agent_core.change_surface.investigation import InvestigationArtifactBuilder
from agent_core.change_surface.model import (
    InvestigationArtifact,
    RepositoryAwareTask,
    ResolvedChangeSurface,
)
from agent_core.change_surface.replan import RepositoryAwareReplanner
from agent_core.change_surface.resolver import ChangeSurfaceResolver
from agent_core.change_surface.scope_guard import PatchScopeGuard
from agent_core.codebase.model import (
    HybridRelevantFileSelection,
    RepositoryFileIndex,
    RepositoryProfile,
)
from agent_core.context.model import TaskContextBundle
from agent_core.coding.patch_generation_model import FilePatchProposal
from agent_core.coding.patch_model import PatchPlan
from agent_core.harness.model import HarnessDecision, HarnessPolicy
from agent_core.harness.trace import HarnessTraceEvent, JsonlHarnessTraceStore
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TaskPlan,
    TechnicalSpec,
)
from agent_core.artifacts.execution_model import (
    TaskExecutionArtifact,
)


class HarnessV1Coordinator:
    """v1.0 控制平面：把模型判断逐层变成 Artifact、Scope 与可验证事实。"""

    def __init__(
        self,
        *,
        store: JsonTaskArtifactStore,
        investigation_builder: InvestigationArtifactBuilder,
        surface_resolver: ChangeSurfaceResolver,
        context_policy: RoleAwareContextPolicy | None = None,
        scope_guard: PatchScopeGuard | None = None,
        policy: HarnessPolicy | None = None,
        trace_store: JsonlHarnessTraceStore | None = None,
    ) -> None:
        self.store = store
        self.artifact_resolver = TaskArtifactResolver(store=store)
        self.investigation_builder = investigation_builder
        self.surface_resolver = surface_resolver
        self.context_policy = context_policy or RoleAwareContextPolicy()
        self.scope_guard = scope_guard or PatchScopeGuard()
        self.policy = policy or HarnessPolicy()
        self.trace_store = trace_store

    def _trace(
        self, *, task_id: str, event_type: str, data: dict | None = None
    ) -> None:
        if self.trace_store is None:
            return
        self.trace_store.append(
            HarnessTraceEvent(
                task_id=task_id,
                event_type=event_type,
                data=data or {},
            )
        )

    async def record_investigation(
        self,
        *,
        bundle: TaskContextBundle,
    ) -> InvestigationArtifact:
        artifact = await self.investigation_builder.build(bundle=bundle)
        self.store.put_model(
            task_id=bundle.task.task_id,
            artifact_type="investigation",
            model=artifact,
            depends_on_task_ids=list(bundle.task.depends_on),
        )
        self._trace(
            task_id=bundle.task.task_id,
            event_type="investigation_recorded",
            data={"observed_file_count": len(artifact.observed_files)},
        )
        return artifact

    async def resolve_change_surface(
        self,
        *,
        bundle: TaskContextBundle,
        selection: HybridRelevantFileSelection,
        task_plan: TaskPlan | None = None,
    ) -> ResolvedChangeSurface:
        investigations = self.artifact_resolver.upstream_investigations(
            task=bundle.task,
            task_plan=task_plan,
        )
        executions = self.artifact_resolver.upstream_executions(
            task=bundle.task,
            task_plan=task_plan,
        )
        expected_investigation_ids = (
            self.artifact_resolver.upstream_investigation_task_ids(
                task=bundle.task,
                task_plan=task_plan,
            )
            if task_plan is not None
            else []
        )
        found_investigation_ids = {item.task_id for item in investigations}
        missing_investigation_ids = sorted(
            set(expected_investigation_ids) - found_investigation_ids
        )

        if (
            self.policy.require_investigation_for_code_change
            and bundle.task.task_type == "code_change"
            and (
                missing_investigation_ids
                or (task_plan is None and bundle.task.depends_on and not investigations)
            )
        ):
            available = [
                f"{item.task_id}:{item.artifact_type}" for item in self.store.list()
            ]
            raise RuntimeError(
                "code_change Task 缺少上游 InvestigationArtifact; "
                f"task={bundle.task.task_id}, "
                f"depends_on={bundle.task.depends_on}, "
                f"expected_investigations={expected_investigation_ids}, "
                f"missing={missing_investigation_ids}, "
                f"available_artifacts={available}"
            )

        surface = await self.surface_resolver.resolve(
            bundle=bundle,
            selection=selection,
            upstream_investigations=investigations,
            upstream_executions=executions,
        )
        if self.policy.fail_on_unresolved_surface and surface.unresolved:
            raise RuntimeError("ResolvedChangeSurface 仍包含 unresolved 项")

        self.store.put_model(
            task_id=bundle.task.task_id,
            artifact_type="change_surface",
            model=surface,
            depends_on_task_ids=list(bundle.task.depends_on),
        )
        self._trace(
            task_id=bundle.task.task_id,
            event_type="surface_resolved",
            data={
                "implementation_targets": list(surface.implementation_targets),
                "dependencies": list(surface.dependencies),
                "references": list(surface.references),
                "tests": list(surface.tests),
                "execution_disposition": surface.execution_disposition,
            },
        )
        return surface

    def build_resolved_context(
        self,
        *,
        repository_root: str | Path,
        requirement: RequirementSpec,
        technical_spec: TechnicalSpec,
        repository: RepositoryProfile,
        index: RepositoryFileIndex,
        bundle: TaskContextBundle,
        selection: HybridRelevantFileSelection,
        surface: ResolvedChangeSurface,
        engineering_rules: list[str] | None = None,
        task_plan: TaskPlan | None = None,
    ) -> TaskContextBundle:
        builder = ResolvedTaskContextBuilder(
            artifact_resolver=self.artifact_resolver,
            context_policy=self.context_policy,
        )
        resolved = builder.build(
            repository_root=repository_root,
            requirement=requirement,
            technical_spec=technical_spec,
            task=bundle.task,
            repository=repository,
            index=index,
            selection=selection,
            surface=surface,
            engineering_rules=engineering_rules,
            task_plan=task_plan,
        )
        self._trace(
            task_id=bundle.task.task_id,
            event_type="context_refined",
            data={
                "context_files": [item.relative_path for item in resolved.code_context]
            },
        )
        return resolved

    def guard_patch_plan(
        self,
        *,
        plan: PatchPlan,
        surface: ResolvedChangeSurface,
    ) -> HarnessDecision:
        result = self.scope_guard.evaluate_plan(plan=plan, surface=surface)
        if (
            self.policy.require_change_surface_before_patch
            and plan.task_id != surface.task_id
        ):
            result = result.model_copy(
                update={
                    "allowed": False,
                    "violations": [*result.violations, "Surface/Plan task_id 不一致"],
                }
            )
        decision = HarnessDecision(
            task_id=plan.task_id,
            allowed=result.allowed,
            stage="patch_scope",
            reasons=result.violations,
        )
        self._trace(
            task_id=plan.task_id,
            event_type="scope_checked",
            data=decision.model_dump(mode="json"),
        )
        return decision

    def guard_patch_proposal(
        self,
        *,
        proposal: FilePatchProposal,
        surface: ResolvedChangeSurface,
    ) -> HarnessDecision:
        result = self.scope_guard.evaluate_proposal(proposal=proposal, surface=surface)
        decision = HarnessDecision(
            task_id=proposal.task_id,
            allowed=result.allowed,
            stage="patch_scope",
            reasons=result.violations,
        )
        self._trace(
            task_id=proposal.task_id,
            event_type="scope_checked",
            data=decision.model_dump(mode="json"),
        )
        return decision

    def replan(
        self,
        *,
        task_plan: TaskPlan,
        repository_revision: str | None = None,
    ):

        replanner = RepositoryAwareReplanner(artifact_resolver=(self.artifact_resolver))

        plan = replanner.build(
            task_plan=task_plan,
            repository_revision=(repository_revision),
        )

        source_task_map = {task.task_id: task for task in task_plan.tasks}

        for task in plan.tasks:

            # ==================================
            # Durable Replan Artifact
            #
            # 一个 Task 的 Repository-aware
            # Resolution 也属于 Artifact。
            #
            # 相同内容重复调用保持幂等。
            # ==================================

            existing_envelope = self.store.latest(
                task_id=task.task_id,
                artifact_type="replan",
            )

            should_publish = True

            if existing_envelope is not None:

                existing = RepositoryAwareTask.model_validate(existing_envelope.payload)

                if existing == task:
                    should_publish = False

            if should_publish:
                source_task = source_task_map[task.task_id]

                self.store.put_model(
                    task_id=(task.task_id),
                    artifact_type="replan",
                    model=task,
                    depends_on_task_ids=list(source_task.depends_on),
                )

            self._trace(
                task_id=task.task_id,
                event_type=("repository_replanned"),
                data={
                    "source_planning_stage": (task.source_planning_stage),
                    "repository_revision": (task.repository_revision),
                    "responsibilities": (list(task.responsibilities)),
                    "original_repository_hints": (list(task.original_repository_hints)),
                    "resolution_status": (task.resolution_status),
                    "execution_disposition": (task.execution_disposition),
                    "resolved_target_files": (list(task.resolved_target_files)),
                },
            )

        return plan

    def record_execution(
        self,
        *,
        task: TaskDefinition,
        artifact: TaskExecutionArtifact,
    ) -> TaskExecutionArtifact:
        """
        把已经通过 Runtime 验证的
        Execution Artifact 写入统一
        Task Artifact Store。

        同一个 Task + Proposal 重复发布时：
            返回已有 Artifact。

        同一个 Task 出现两个不同 Proposal：
            直接失败，避免状态歧义。
        """

        if artifact.task_id != task.task_id:
            raise ValueError("Execution Artifact " "task_id 与 Task 不一致")

        existing_envelope = self.store.latest(
            task_id=task.task_id,
            artifact_type="execution",
        )

        if existing_envelope is not None:
            existing = TaskExecutionArtifact.model_validate(existing_envelope.payload)

            if existing.proposal_id != artifact.proposal_id:
                raise RuntimeError(
                    "同一个 Task 已存在不同 "
                    "Proposal 的 Execution Artifact; "
                    f"task={task.task_id}, "
                    f"existing={existing.proposal_id}, "
                    f"incoming={artifact.proposal_id}"
                )

            if existing != artifact:
                raise RuntimeError(
                    "同一个 Proposal 对应的 " "Execution Artifact 内容不一致"
                )

            # Idempotent retry。
            return existing

        self.store.put_model(
            task_id=task.task_id,
            artifact_type="execution",
            model=artifact,
            depends_on_task_ids=list(task.depends_on),
        )

        self._trace(
            task_id=task.task_id,
            event_type="execution_recorded",
            data={
                "proposal_id": (artifact.proposal_id),
                "base_revision": (artifact.base_revision),
                "result_revision": (artifact.result_revision),
                "changed_files": [
                    item.relative_path for item in artifact.changed_files
                ],
                "verification_checks": [
                    item.name for item in artifact.verification_checks
                ],
            },
        )

        return artifact

    def attach_upstream_artifacts(
        self,
        *,
        bundle: TaskContextBundle,
        task_plan: TaskPlan | None = None,
    ) -> TaskContextBundle:
        """
        把上游持久化 Artifact 放进
        当前 TaskContextBundle。

        这样 Investigation / Coding /
        Renderer 都可以看到真实历史状态。
        """

        investigations = self.artifact_resolver.upstream_investigations(
            task=bundle.task,
            task_plan=task_plan,
        )

        executions = self.artifact_resolver.upstream_executions(
            task=bundle.task,
            task_plan=task_plan,
        )

        upstream = [item.model_dump(mode="json") for item in investigations]

        upstream.extend(item.model_dump(mode="json") for item in executions)

        return bundle.model_copy(
            deep=True,
            update={
                "upstream_artifacts": upstream,
            },
        )
