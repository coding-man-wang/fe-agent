from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from uuid import uuid4

from agent_core.benchmark.acceptance import HiddenAcceptanceRunner
from agent_core.benchmark.committed_snapshot import CommittedSnapshotWorktreeManager
from agent_core.benchmark.case_model import BenchmarkCase
from agent_core.benchmark.execution_decision import RuntimeExecutionDecisionPolicy
from agent_core.benchmark.execution_evaluator import FullExecutionEvaluator
from agent_core.benchmark.execution_model import (
    FullExecutionRunRecord,
    FullExecutionTaskRecord,
)
from agent_core.benchmark.execution_store import FullExecutionRunStore
from agent_core.benchmark.execution_surface import ProductionExecutionSurfacePolicy
from agent_core.benchmark.git_diff import GitDiffCollector
from agent_core.benchmark.git_workspace import RunScopedGitWorkspaceManager
from agent_core.benchmark.text_model import LLMTextModelAdapter
from agent_core.benchmark.verification_policy import build_frontend_verification_policy
from agent_core.benchmark.workspace_dependencies import NodeModulesWorkspaceProvisioner
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.codebase.model import RepositoryProfile
from agent_core.codebase.task_context_selection import (
    TaskContextSelectionService,
)
from agent_core.coding.agent import ReadOnlyCodingAgent
from agent_core.coding.approval_store import ApprovalStore
from agent_core.coding.completion_store import TaskCompletionStore
from agent_core.coding.execution_journal import ExecutionJournalStore
from agent_core.coding.handoff import AnalysisArtifactStore, calculate_messages_hash
from agent_core.coding.orchestration_model import TaskAnalysisRecord
from agent_core.coding.patch_generator import PatchGenerator
from agent_core.coding.patch_planner import PatchPlanner
from agent_core.coding.preparation import ExecutionPreparer
from agent_core.coding.reviewer import PatchReviewer
from agent_core.coding.snapshot import FileSnapshotBuilder
from agent_core.coding.verification_plan import VerificationPlan
from agent_core.coding.verification_store import VerificationStore
from agent_core.coding.workspace_execution import WorkspaceExecutionService
from agent_core.git.commit_service import AgentGitCommitService
from agent_core.git.workspace_store import GitWorkspaceStore
from agent_core.harness.coordinator import HarnessV1Coordinator
from agent_core.llm.base import LLM
from agent_core.sdd.model import RequirementSpec, TaskPlan, TechnicalSpec
from agent_core.sdd.task_graph import topological_sort_tasks
from agent_core.artifacts.store import (
    JsonTaskArtifactStore,
)
from agent_core.coding.execution_artifact import (
    TaskExecutionArtifactBuilder,
)
from agent_core.harness.trace import (
    JsonlHarnessTraceStore,
)
from agent_core.codebase.task_context_selection import (
    TaskContextSelectionService,
)
from agent_core.codebase.impact import (
    RepositoryChangeImpactAnalyzer,
)
from agent_core.benchmark.impact_verification import (
    ImpactAwareVerificationPlanner,
)


class FullExecutionBenchmarkError(RuntimeError):
    pass


class FullExecutionBenchmarkService:
    """v1.1: execute production code-change tasks through the real side-effect plane."""

    def __init__(
        self,
        *,
        fe_agent_root: str | Path,
        benchmark_repository: str | Path,
        baseline_run_id: str,
        llm: LLM,
        coordinator: HarnessV1Coordinator,
        context_selection_service: TaskContextSelectionService,
    ) -> None:
        self.fe_agent_root = Path(fe_agent_root).resolve(strict=True)
        self.benchmark_repository = Path(benchmark_repository).resolve(strict=True)
        self.baseline_run_id = baseline_run_id
        self.llm = llm
        self.coordinator = coordinator
        self.run_id = uuid4().hex
        self.run_root = (
            self.fe_agent_root
            / ".fe-agent"
            / "benchmark"
            / "full_execution"
            / self.run_id
        )
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root = self.run_root / "worktrees"
        self.snapshot_root = self.run_root / "committed_snapshots"
        self.runtime_state_root = self.run_root / "state"
        self.runtime_state_root.mkdir(parents=True, exist_ok=True)
        self.store = FullExecutionRunStore(self.fe_agent_root)
        self.context_selection_service = context_selection_service

    def _create_run_scoped_coordinator(
        self,
    ) -> HarnessV1Coordinator:
        """
        为当前 Full Execution Run 创建
        独立 Harness Artifact Store。

        baseline coordinator 只作为历史证据来源。

        本次执行产生的 execution artifact
        只能写入：

            full_execution/<run_id>/harness

        防止污染 historical baseline。
        """

        harness_root = self.run_root / "harness"

        harness_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        run_store = JsonTaskArtifactStore(root=(harness_root / "artifacts"))

        # 只复制静态/历史分析证据。
        #
        # 特别注意：
        # execution 不允许从 baseline 复制。
        #
        # 否则可能把另一次 Full Execution
        # 的执行事实带进当前 Run。
        run_store.seed_from(
            source=self.coordinator.store,
            artifact_types={
                "investigation",
                "change_surface",
                "replan",
            },
        )

        return HarnessV1Coordinator(
            store=run_store,
            investigation_builder=(self.coordinator.investigation_builder),
            surface_resolver=(self.coordinator.surface_resolver),
            context_policy=(self.coordinator.context_policy),
            scope_guard=(self.coordinator.scope_guard),
            policy=(self.coordinator.policy),
            trace_store=(JsonlHarnessTraceStore(path=(harness_root / "trace.jsonl"))),
        )

    async def run(
        self,
        *,
        case: BenchmarkCase,
        task_plan: TaskPlan,
        requirement: RequirementSpec,
        technical_spec: TechnicalSpec,
        repository: RepositoryProfile,
        engineering_rules: list[str],
        task_surfaces: dict[str, ResolvedChangeSurface],
    ) -> FullExecutionRunRecord:
        record = FullExecutionRunRecord(
            run_id=self.run_id,
            baseline_run_id=self.baseline_run_id,
            case_id=case.case_id,
            base_commit_sha=case.base_commit_sha,
        )
        self.store.save(record)

        run_coordinator = self._create_run_scoped_coordinator()

        text_model = LLMTextModelAdapter(llm=self.llm)
        coding_agent = ReadOnlyCodingAgent(model=text_model)
        analysis_store = AnalysisArtifactStore(self.runtime_state_root)
        approval_store = ApprovalStore(self.runtime_state_root)
        journal_store = ExecutionJournalStore(self.runtime_state_root)
        verification_store = VerificationStore(self.runtime_state_root)
        completion_store = TaskCompletionStore(self.runtime_state_root)
        verification_policy = build_frontend_verification_policy()

        impact_verification_planner = ImpactAwareVerificationPlanner()

        reviewer = PatchReviewer()
        workspace_manager = RunScopedGitWorkspaceManager(
            run_id=self.run_id,
            repository_root=self.benchmark_repository,
            workspace_root=self.workspace_root,
            store=GitWorkspaceStore(self.runtime_state_root),
        )
        dependency_provisioner = NodeModulesWorkspaceProvisioner(
            dependency_source_root=self.benchmark_repository
        )
        commit_service = AgentGitCommitService()
        snapshot_manager = CommittedSnapshotWorktreeManager(
            repository_root=self.benchmark_repository,
            snapshot_root=self.snapshot_root,
        )
        execution_surface_policy = ProductionExecutionSurfacePolicy()
        runtime_execution_policy = RuntimeExecutionDecisionPolicy()

        impact_analyzer = RepositoryChangeImpactAnalyzer(max_depth=2)

        current_root = self.benchmark_repository
        current_sha = case.base_commit_sha
        task_records: list[FullExecutionTaskRecord] = []

        try:
            for task in topological_sort_tasks(task_plan):
                if task.task_type != "code_change":
                    task_records.append(
                        FullExecutionTaskRecord(
                            task_id=task.task_id,
                            task_type=task.task_type,
                            status="skipped_non_code_task",
                        )
                    )
                    continue

                surface = task_surfaces.get(task.task_id)
                if surface is None:
                    raise FullExecutionBenchmarkError(
                        f"{task.task_id} 缺少 ResolvedChangeSurface"
                    )

                if surface.execution_disposition == "no_change_required":
                    task_records.append(
                        FullExecutionTaskRecord(
                            task_id=task.task_id,
                            task_type=task.task_type,
                            status="skipped_no_change",
                            execution_disposition=surface.execution_disposition,
                            base_commit_sha=current_sha,
                        )
                    )
                    continue

                if surface.execution_disposition == "insufficient_evidence":
                    blocked = FullExecutionTaskRecord(
                        task_id=task.task_id,
                        task_type=task.task_type,
                        status="failed",
                        execution_disposition=surface.execution_disposition,
                        base_commit_sha=current_sha,
                        error="InsufficientEvidence: Change Surface 证据不足，禁止进入 Side Effect Plane",
                    )
                    task_records.append(blocked)
                    raise FullExecutionBenchmarkError(blocked.error)

                # v1.1 RC only executes production-code changes. The task-level
                # Change Surface can still contain test files because tests are useful
                # evidence and may be authored in a later phase. Convert them to
                # read-only references *before* building Coding Context so the model
                # and the runtime share the same execution boundary.
                execution_surface = execution_surface_policy.apply(surface)

                if not execution_surface.requires_patch():
                    task_records.append(
                        FullExecutionTaskRecord(
                            task_id=task.task_id,
                            task_type=task.task_type,
                            status="skipped_no_production_change",
                            execution_disposition=surface.execution_disposition,
                            base_commit_sha=current_sha,
                        )
                    )
                    continue

                task_record = FullExecutionTaskRecord(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    status="failed",
                    execution_disposition=surface.execution_disposition,
                    base_commit_sha=current_sha,
                )
                analysis_root: Path | None = None
                try:
                    analysis_root = await snapshot_manager.create(
                        task_id=task.task_id,
                        commit_sha=current_sha,
                    )

                    # ======================================
                    # v1.2.3
                    #
                    # 每个 Task 都根据当前最新 Commit
                    # 重新计算：
                    #
                    # 1. File Index
                    # 2. Retrieval
                    # 3. Semantic Index
                    # 4. Dependency Graph
                    # 5. Final Selection
                    #
                    # 不再复用 Prepared Input 中
                    # 最开始生成的 task_selections。
                    # ======================================

                    context_selection = await self.context_selection_service.select(
                        repository_root=analysis_root,
                        repository_revision=current_sha,
                        task=task,
                        raw_requirement=case.requirement,
                        requirement_spec=requirement,
                    )

                    retrieval_paths = [
                        item.relative_path
                        for item in (
                            context_selection.retrieval_selection.selected_files
                        )
                    ]

                    selected_paths = [
                        item.relative_path
                        for item in (context_selection.selection.selected_files)
                    ]

                    retrieval_path_set = set(retrieval_paths)

                    graph_added_paths = [
                        path
                        for path in selected_paths
                        if path not in retrieval_path_set
                    ]

                    task_record = task_record.model_copy(
                        deep=True,
                        update={
                            "context_revision": (current_sha),
                            "context_retrieval_mode": (
                                context_selection.retrieval_mode
                            ),
                            "semantic_fallback_reason": (
                                context_selection.semantic_error
                            ),
                            "retrieval_files": (retrieval_paths),
                            "selected_context_files": (selected_paths),
                            "graph_added_files": (graph_added_paths),
                        },
                    )
                    current_index = context_selection.index

                    selection = context_selection.selection
                    current_repository = repository.model_copy(
                        deep=True,
                        update={"root_path": str(analysis_root)},
                    )
                    bundle = run_coordinator.build_resolved_context(
                        repository_root=analysis_root,
                        requirement=requirement,
                        technical_spec=technical_spec,
                        repository=current_repository,
                        index=current_index,
                        bundle=self._base_bundle_for_task(
                            task=task,
                            requirement=requirement,
                            technical_spec=technical_spec,
                            repository=current_repository,
                            current_root=analysis_root,
                            selection=selection,
                            engineering_rules=engineering_rules,
                        ),
                        selection=selection,
                        surface=execution_surface,
                        engineering_rules=engineering_rules,
                        task_plan=task_plan,
                    )

                    messages = coding_agent.prepare_messages(bundle)
                    context_hash = calculate_messages_hash(messages)
                    analysis = await coding_agent.analyze_prepared(
                        bundle=bundle,
                        messages=messages,
                    )
                    analysis_record = TaskAnalysisRecord(
                        task_id=task.task_id,
                        status="succeeded",
                        attempts=1,
                        result=analysis,
                        context_hash=context_hash,
                    )
                    artifact, _ = analysis_store.publish(
                        bundle=bundle,
                        record=analysis_record,
                        context_hash=context_hash,
                    )

                    # Fresh-code analysis is allowed to invalidate an earlier
                    # planning hypothesis. A downstream task can become redundant
                    # after upstream verified commits already implemented the
                    # required behavior. Do not force an empty PatchPlan into
                    # PatchGenerator. Distinguish a confirmed runtime no-op from
                    # missing evidence so uncertainty never silently becomes success.
                    runtime_decision = runtime_execution_policy.decide(
                        analysis,
                        surface=execution_surface,
                    )

                    if runtime_decision.action == "skip_runtime_noop":
                        task_record = task_record.model_copy(
                            deep=True,
                            update={
                                "status": "skipped_runtime_noop",
                                "error": None,
                            },
                        )
                        task_records.append(task_record)
                        record = record.model_copy(
                            deep=True,
                            update={"tasks": task_records},
                        )
                        self.store.save(record)
                        continue

                    if runtime_decision.action == "block_insufficient_evidence":
                        raise FullExecutionBenchmarkError(
                            "RuntimeInsufficientEvidence: " + runtime_decision.reason
                        )

                    snapshot = FileSnapshotBuilder(analysis_root).build(bundle)
                    patch_plan = PatchPlanner().plan(
                        artifact=artifact,
                        snapshot=snapshot,
                    )

                    # Defensive invariant: after the runtime decision says
                    # 'continue', PatchPlan must contain at least one target.
                    # This catches future schema/adapter regressions before the
                    # LLM-backed PatchGenerator is invoked.
                    if not patch_plan.targets:
                        raise FullExecutionBenchmarkError(
                            "Runtime invariant violated: continue decision produced "
                            "an empty PatchPlan"
                        )

                    plan_decision = run_coordinator.guard_patch_plan(
                        plan=patch_plan,
                        surface=execution_surface,
                    )
                    if not plan_decision.allowed:
                        raise FullExecutionBenchmarkError(
                            "; ".join(plan_decision.reasons)
                        )

                    preparation = ExecutionPreparer(
                        repository_root=analysis_root,
                        artifact_store=analysis_store,
                        message_provider=coding_agent,
                    ).prepare(
                        task_id=task.task_id,
                        artifact_id=artifact.artifact_id,
                        bundle=bundle,
                    )
                    proposal = await PatchGenerator(
                        repository_root=analysis_root,
                        model=text_model,
                    ).generate(
                        artifact=artifact,
                        plan=patch_plan,
                        preparation=preparation,
                    )
                    proposal_decision = run_coordinator.guard_patch_proposal(
                        proposal=proposal,
                        surface=execution_surface,
                    )
                    if not proposal_decision.allowed:
                        raise FullExecutionBenchmarkError(
                            "; ".join(proposal_decision.reasons)
                        )
                    planned_changed_files = sorted(
                        patch.relative_path for patch in proposal.patches
                    )

                    planned_impact = impact_analyzer.analyze(
                        changed_files=(planned_changed_files),
                        graph=(context_selection.graph),
                        index=current_index,
                        repository_revision=(current_sha),
                        # 使用原始 surface，
                        # 而不是 execution_surface。
                        #
                        # 因为 ProductionExecutionSurfacePolicy
                        # 会把 tests 转成只读 references。
                        #
                        # 但做影响分析时，
                        # 我们仍然需要知道哪些测试是
                        # Change Surface 已经识别出来的。
                        known_test_files=list(surface.tests),
                    )

                    planned_verification = impact_verification_planner.build(
                        base_command_ids=list(
                            case.expectation.verification_command_ids
                        ),
                        impact=planned_impact,
                    )

                    task_record = task_record.model_copy(
                        deep=True,
                        update={
                            "planned_impact": (planned_impact),
                            "planned_verification": (planned_verification),
                        },
                    )
                    review = reviewer.prepare(proposal)
                    task_record = task_record.model_copy(
                        deep=True,
                        update={
                            "proposal_id": review.proposal_id,
                            "proposed_files": sorted(
                                patch.relative_path for patch in proposal.patches
                            ),
                        },
                    )
                    approval_store.record_decision(
                        review=review,
                        decision="approved",
                        reason=(
                            "benchmark auto-approval: schema + evidence + "
                            "change-surface scope guard passed"
                        ),
                    )
                    workspace = await workspace_manager.create(
                        task_id=task.task_id,
                        proposal_id=review.proposal_id,
                        base_commit_sha=current_sha,
                    )
                    task_record = task_record.model_copy(
                        deep=True,
                        update={"workspace_id": workspace.workspace_id},
                    )
                    self._assert_proposal_matches_execution_workspace(
                        proposal=proposal,
                        execution_root=Path(workspace.worktree_path),
                        analysis_root=analysis_root,
                        commit_sha=current_sha,
                    )
                    dependency_provisioner.provision(workspace.worktree_path)

                    verification_plan = VerificationPlan(
                        proposal_id=review.proposal_id,
                        task_id=task.task_id,
                        command_ids=list(planned_verification.final_command_ids),
                    )
                    execution = await WorkspaceExecutionService(
                        runtime_root=self.runtime_state_root,
                        approval_store=approval_store,
                        journal_store=journal_store,
                        verification_store=verification_store,
                        completion_store=completion_store,
                        verification_policy=verification_policy,
                    ).execute(
                        workspace=workspace,
                        proposal=proposal,
                        verification_plan=verification_plan,
                    )
                    commit = await commit_service.commit(
                        workspace=workspace,
                        message=f"benchmark({case.case_id}): {task.task_id} {task.title}",
                    )

                    execution_artifact = TaskExecutionArtifactBuilder.build(
                        journal=execution.journal,
                        completion=execution.completion,
                        verification=(execution.verification),
                        base_revision=current_sha,
                        result_revision=(commit.commit_sha),
                    )

                    run_coordinator.record_execution(
                        task=task,
                        artifact=execution_artifact,
                    )

                    task_record = task_record.model_copy(
                        deep=True,
                        update={
                            "status": "completed",
                            "workspace_id": workspace.workspace_id,
                            "proposal_id": review.proposal_id,
                            "proposed_files": sorted(
                                patch.relative_path for patch in proposal.patches
                            ),
                            "verification_status": execution.verification.status,
                            "completion_status": execution.completion.status,
                            "commit_sha": commit.commit_sha,
                            "error": None,
                        },
                    )
                    current_root = Path(workspace.worktree_path).resolve(strict=True)
                    current_sha = commit.commit_sha
                except Exception as exc:
                    error_message = f"{type(exc).__name__}: {exc}"
                    verification_status = task_record.verification_status
                    verification_checks = list(task_record.verification_checks)

                    # WorkspaceExecutionService intentionally raises a compact
                    # gate error after Verification fails. The durable
                    # VerificationRunRecord already contains the real command
                    # result (name/status/stdout/stderr), so surface that
                    # evidence in the benchmark Task record instead of hiding
                    # it behind "Verification 未成功".
                    if task_record.proposal_id and verification_store.exists(
                        task_record.proposal_id
                    ):
                        verification_record = verification_store.load(
                            task_record.proposal_id
                        )
                        verification_status = verification_record.status

                        if verification_record.result is not None:
                            verification_checks = list(
                                verification_record.result.checks
                            )
                            failed_checks = [
                                check
                                for check in verification_checks
                                if check.status == "failed"
                            ]
                            if failed_checks:
                                details = "\n\n".join(
                                    f"[{check.name}] {check.status}\n{check.details}"
                                    for check in failed_checks
                                )
                                error_message += (
                                    "\nVerification failed checks:\n" + details
                                )
                        elif verification_record.error:
                            error_message += (
                                "\nVerification workflow error: "
                                + verification_record.error
                            )

                    task_record = task_record.model_copy(
                        deep=True,
                        update={
                            "verification_status": verification_status,
                            "verification_checks": verification_checks,
                            "error": error_message,
                        },
                    )
                    task_records.append(task_record)
                    raise
                finally:
                    if analysis_root is not None:
                        await snapshot_manager.remove(
                            analysis_root,
                            allow_missing=True,
                        )

                task_records.append(task_record)
                record = record.model_copy(deep=True, update={"tasks": task_records})
                self.store.save(record)

            hidden = await HiddenAcceptanceRunner(
                fixture_root=self.fe_agent_root / "benchmark" / "acceptance"
            ).run(
                case_id=case.case_id,
                repository_root=current_root,
            )
            actual_files = await GitDiffCollector().changed_files(
                repository_root=self.benchmark_repository,
                base_commit_sha=case.base_commit_sha,
                head_commit_sha=current_sha,
            )
            evaluation = FullExecutionEvaluator().evaluate(
                case=case,
                actual_change_files=actual_files,
                task_records=task_records,
                hidden_acceptance=hidden,
            )
            completed = record.model_copy(
                deep=True,
                update={
                    "status": "completed",
                    "finished_at": datetime.now(timezone.utc),
                    "final_commit_sha": current_sha,
                    "final_workspace_root": str(current_root),
                    "tasks": task_records,
                    "hidden_acceptance": hidden,
                    "evaluation": evaluation,
                    "error": None,
                },
            )
            self.store.save(completed)
            return completed
        except Exception as exc:
            failed = record.model_copy(
                deep=True,
                update={
                    "status": "failed",
                    "finished_at": datetime.now(timezone.utc),
                    "final_commit_sha": current_sha,
                    "final_workspace_root": str(current_root),
                    "tasks": task_records,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            self.store.save(failed)
            return failed

    @staticmethod
    def _assert_proposal_matches_execution_workspace(
        *,
        proposal,
        execution_root: Path,
        analysis_root: Path,
        commit_sha: str,
    ) -> None:
        """Fail before Apply if analysis and execution roots diverged."""
        for patch in proposal.patches:
            path = execution_root / patch.relative_path
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise FullExecutionBenchmarkError(
                    "Execution workspace 无法读取 Proposal 目标: "
                    f"{patch.relative_path}"
                ) from exc
            if actual != patch.expected_sha256:
                raise FullExecutionBenchmarkError(
                    "Proposal base 与 execution workspace 不一致: "
                    f"path={patch.relative_path}, commit={commit_sha}, "
                    f"expected={patch.expected_sha256}, actual={actual}, "
                    f"analysis_root={analysis_root}, "
                    f"execution_root={execution_root}"
                )

    @staticmethod
    def _base_bundle_for_task(
        *,
        task,
        requirement,
        technical_spec,
        repository,
        current_root,
        selection,
        engineering_rules,
    ):
        # Import here to keep benchmark module dependencies one-directional.
        from agent_core.context.builder import ProjectContextBuilder

        return ProjectContextBuilder().build(
            repository_root=current_root,
            requirement=requirement,
            technical_spec=technical_spec,
            task=task,
            repository=repository,
            selection=selection,
            engineering_rules=engineering_rules,
            max_files=8,
            max_total_chars=40_000,
            max_chars_per_file=10_000,
        )
