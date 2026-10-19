from pathlib import Path

from agent_core.artifacts.resolver import TaskArtifactResolver
from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.change_surface.context_builder import ResolvedTaskContextBuilder
from agent_core.change_surface.context_policy import RoleAwareContextPolicy
from agent_core.change_surface.investigation import InvestigationArtifactBuilder
from agent_core.change_surface.model import InvestigationArtifact, ResolvedChangeSurface
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
from agent_core.sdd.model import RequirementSpec, TaskPlan, TechnicalSpec


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

    def _trace(self, *, task_id: str, event_type: str, data: dict | None = None) -> None:
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
                or (
                    task_plan is None
                    and bundle.task.depends_on
                    and not investigations
                )
            )
        ):
            available = [
                f"{item.task_id}:{item.artifact_type}"
                for item in self.store.list()
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
            data={"context_files": [item.relative_path for item in resolved.code_context]},
        )
        return resolved

    def guard_patch_plan(
        self,
        *,
        plan: PatchPlan,
        surface: ResolvedChangeSurface,
    ) -> HarnessDecision:
        result = self.scope_guard.evaluate_plan(plan=plan, surface=surface)
        if self.policy.require_change_surface_before_patch and plan.task_id != surface.task_id:
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

    def replan(self, *, task_plan: TaskPlan):
        replanner = RepositoryAwareReplanner(artifact_resolver=self.artifact_resolver)
        plan = replanner.build(task_plan=task_plan)
        for task in plan.tasks:
            self._trace(
                task_id=task.task_id,
                event_type="repository_replanned",
                data={
                    "resolution_status": task.resolution_status,
                    "execution_disposition": task.execution_disposition,
                    "resolved_target_files": list(task.resolved_target_files),
                },
            )
        return plan
