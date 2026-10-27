from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from uuid import uuid4

from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.benchmark.acceptance import HiddenAcceptanceRunner
from agent_core.benchmark.case_model import BenchmarkCase
from agent_core.benchmark.committed_snapshot import (
    CommittedSnapshotWorktreeManager,
)
from agent_core.benchmark.execution_decision import (
    RuntimeExecutionDecisionPolicy,
)
from agent_core.benchmark.execution_evaluator import (
    FullExecutionEvaluator,
)
from agent_core.benchmark.execution_model import (
    FullExecutionRunRecord,
    FullExecutionTaskRecord,
)
from agent_core.benchmark.execution_store import (
    FullExecutionRunStore,
)
from agent_core.benchmark.execution_surface import (
    ProductionExecutionSurfacePolicy,
    TestExecutionSurfacePolicy,
)
from agent_core.benchmark.final_verification import (
    FinalVerificationPlanner,
)
from agent_core.benchmark.git_diff import GitDiffCollector
from agent_core.benchmark.git_workspace import (
    RunScopedGitWorkspaceManager,
)
from agent_core.benchmark.impact_verification import (
    ImpactAwareVerificationPlanner,
)
from agent_core.benchmark.task_verification import (
    TaskVerificationCommandPolicy,
)
from agent_core.benchmark.text_model import (
    LLMTextModelAdapter,
)
from agent_core.benchmark.verification_policy import (
    build_frontend_verification_policy,
)
from agent_core.benchmark.workspace_dependencies import (
    NodeModulesWorkspaceProvisioner,
)
from agent_core.change_surface.model import (
    ResolvedChangeSurface,
)
from agent_core.codebase.impact import (
    RepositoryChangeImpactAnalyzer,
)
from agent_core.codebase.model import RepositoryProfile
from agent_core.codebase.task_context_selection import (
    TaskContextSelectionService,
)
from agent_core.coding.agent import ReadOnlyCodingAgent
from agent_core.coding.approval_store import ApprovalStore
from agent_core.coding.completion_store import (
    TaskCompletionStore,
)
from agent_core.coding.execution_artifact import (
    TaskExecutionArtifactBuilder,
)
from agent_core.coding.execution_journal import (
    ExecutionJournalStore,
)
from agent_core.coding.handoff import (
    AnalysisArtifactStore,
    calculate_messages_hash,
)
from agent_core.coding.orchestration_model import (
    TaskAnalysisRecord,
)
from agent_core.coding.patch_generator import PatchGenerator, PatchGenerationNoop
from agent_core.coding.patch_planner import PatchPlanner
from agent_core.coding.preparation import ExecutionPreparer
from agent_core.coding.reviewer import PatchReviewer
from agent_core.coding.snapshot import FileSnapshotBuilder
from agent_core.coding.verification_executor import (
    VerificationExecutor,
)
from agent_core.coding.verification_plan import (
    VerificationPlan,
)
from agent_core.coding.verification_store import (
    VerificationStore,
)
from agent_core.coding.workspace_execution import (
    WorkspaceExecutionService,
)
from agent_core.git.commit_service import (
    AgentGitCommitService,
)
from agent_core.git.workspace_store import GitWorkspaceStore
from agent_core.harness.coordinator import (
    HarnessV1Coordinator,
)
from agent_core.harness.trace import (
    JsonlHarnessTraceStore,
)
from agent_core.llm.base import LLM
from agent_core.sdd.model import (
    RequirementSpec,
    TaskPlan,
    TechnicalSpec,
)
from agent_core.sdd.task_graph import (
    topological_sort_tasks,
)


class FullExecutionBenchmarkError(RuntimeError):
    pass


class FullExecutionBenchmarkService:
    """
    Full Execution Benchmark。

    当前执行模型：

        investigation
            ↓
        skip（历史分析证据已由 Prepared Input 提供）

        code_change
            ↓
        production-only Side Effect
            ↓
        staged structural verification
            ↓
        commit

        test
            ↓
        test-only Side Effect
            ↓
        staged unit verification
            ↓
        commit

        verification
            ↓
        final run-level verification
            ↓
        不修改代码、不产生 commit

        hidden acceptance
            ↓
        final evaluation
    """

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

        self.run_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.workspace_root = self.run_root / "worktrees"

        self.snapshot_root = self.run_root / "committed_snapshots"

        self.runtime_state_root = self.run_root / "state"

        self.runtime_state_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.store = FullExecutionRunStore(self.fe_agent_root)

        self.context_selection_service = context_selection_service

    def _create_run_scoped_coordinator(
        self,
    ) -> HarnessV1Coordinator:
        """
        为当前 Full Execution Run 创建
        独立 Harness Artifact Store。

        baseline coordinator 只作为
        历史证据来源。

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

        # 只复制静态 / 历史分析证据。
        #
        # execution 不允许从 baseline
        # 复制，否则会污染本次 Run 的
        # 真实执行事实。
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
        task_surfaces: dict[
            str,
            ResolvedChangeSurface,
        ],
    ) -> FullExecutionRunRecord:

        record = FullExecutionRunRecord(
            run_id=self.run_id,
            baseline_run_id=(self.baseline_run_id),
            case_id=case.case_id,
            base_commit_sha=(case.base_commit_sha),
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

        final_verification_planner = FinalVerificationPlanner()

        task_verification_policy = TaskVerificationCommandPolicy()

        production_surface_policy = ProductionExecutionSurfacePolicy()

        test_surface_policy = TestExecutionSurfacePolicy()

        runtime_execution_policy = RuntimeExecutionDecisionPolicy()

        impact_analyzer = RepositoryChangeImpactAnalyzer(max_depth=2)

        reviewer = PatchReviewer()

        workspace_manager = RunScopedGitWorkspaceManager(
            run_id=self.run_id,
            repository_root=(self.benchmark_repository),
            workspace_root=(self.workspace_root),
            store=GitWorkspaceStore(self.runtime_state_root),
        )

        dependency_provisioner = NodeModulesWorkspaceProvisioner(
            dependency_source_root=(self.benchmark_repository)
        )

        commit_service = AgentGitCommitService()

        snapshot_manager = CommittedSnapshotWorktreeManager(
            repository_root=(self.benchmark_repository),
            snapshot_root=(self.snapshot_root),
        )

        current_root = self.benchmark_repository

        current_sha = case.base_commit_sha

        task_records: list[FullExecutionTaskRecord] = []

        completed_final_verification_keys: set[
            tuple[
                str,
                tuple[str, ...],
            ]
        ] = set()

        try:
            for task in topological_sort_tasks(task_plan):
                # ==================================
                # Investigation
                # ==================================

                if task.task_type == "investigation":
                    task_records.append(
                        FullExecutionTaskRecord(
                            task_id=(task.task_id),
                            task_type=(task.task_type),
                            status=("skipped_non_code_task"),
                        )
                    )

                    record = record.model_copy(
                        deep=True,
                        update={"tasks": (task_records)},
                    )

                    self.store.save(record)

                    continue

                # ==================================
                # Final Verification
                #
                # 不经过 Coding Agent。
                # 不产生 Patch。
                # 不产生 Commit。
                #
                # 验证当前最新 Revision。
                # ==================================

                if task.task_type == "verification":
                    final_plan = final_verification_planner.build(
                        base_command_ids=list(
                            case.expectation.verification_command_ids
                        ),
                        task_records=(task_records),
                    )

                    verification_key = (
                        current_sha,
                        tuple(final_plan.final_command_ids),
                    )

                    if verification_key in completed_final_verification_keys:
                        task_record = FullExecutionTaskRecord(
                            task_id=task.task_id,
                            task_type=task.task_type,
                            status=("skipped_redundant_verification"),
                            base_commit_sha=current_sha,
                            context_revision=current_sha,
                            planned_verification=(final_plan),
                        )

                        task_records.append(task_record)

                        record = record.model_copy(
                            deep=True,
                            update={"tasks": task_records},
                        )

                        self.store.save(record)

                        continue

                    verification_run_id = hashlib.sha256(
                        (
                            f"{case.case_id}:"
                            f"{task.task_id}:"
                            f"{current_sha}:"
                            "final-verification-v1"
                        ).encode("utf-8")
                    ).hexdigest()

                    # current_root 是最后一个成功
                    # Side Effect Task 对应的
                    # 最新 Worktree。
                    #
                    # node_modules 已存在时这里只做
                    # readiness validation。
                    dependency_provisioner.provision(current_root)

                    verification_plan = VerificationPlan(
                        proposal_id=(verification_run_id),
                        task_id=(task.task_id),
                        command_ids=list(final_plan.final_command_ids),
                    )

                    executor = VerificationExecutor(
                        repository_root=(current_root),
                        policy=(verification_policy),
                    )

                    result = await executor.run(plan=(verification_plan))

                    executed_command_ids = [
                        check.command_id
                        for check in result.checks
                        if (check.command_id is not None)
                    ]

                    task_record = FullExecutionTaskRecord(
                        task_id=(task.task_id),
                        task_type=(task.task_type),
                        status=(
                            "completed" if (result.status == "passed") else "failed"
                        ),
                        base_commit_sha=(current_sha),
                        context_revision=(current_sha),
                        planned_verification=(final_plan),
                        scheduled_verification_command_ids=list(
                            final_plan.final_command_ids
                        ),
                        executed_verification_command_ids=(executed_command_ids),
                        verification_status=(
                            "succeeded" if (result.status == "passed") else "failed"
                        ),
                        verification_checks=list(result.checks),
                        error=(
                            None
                            if (result.status == "passed")
                            else ("Final Verification " "failed")
                        ),
                    )

                    task_records.append(task_record)

                    record = record.model_copy(
                        deep=True,
                        update={"tasks": (task_records)},
                    )

                    self.store.save(record)

                    if result.status != "passed":
                        raise (
                            FullExecutionBenchmarkError("Final Verification " "failed")
                        )

                    self.store.save(record)

                    if result.status != "passed":
                        raise (
                            FullExecutionBenchmarkError("Final Verification " "failed")
                        )

                    # Final Verification 已成功。
                    #
                    # 记录：
                    # 当前 Revision + 当前 Verification Commands
                    # 已经完成过最终验证。
                    completed_final_verification_keys.add(verification_key)

                    continue

                # ==================================
                # Side Effect Task
                #
                # 当前支持：
                #
                # code_change
                # test
                #
                # 重要：
                #
                # task_surfaces 是 Full Execution
                # 开始前生成的 Prepared / Baseline
                # 分析结果。
                #
                # Repository 在多 Task 执行过程中
                # 会不断产生新的 Commit，
                # 因此不能直接使用旧 Surface
                # 作为当前 Task 的最终执行判断。
                #
                # 当前 Task 必须：
                #
                # latest commit
                #     ↓
                # fresh context
                #     ↓
                # fresh change surface
                #     ↓
                # side effect decision
                # ==================================

                # 目前仍保留 task_surfaces 参数，
                # 用来验证 Prepared Input 完整性。
                #
                # 但它不再决定：
                #
                # - 当前是否需要修改
                # - 当前修改哪些文件
                # - 当前是否 evidence insufficient
                #
                # 后续可以单独清理这个历史参数。
                if task.task_id not in task_surfaces:
                    raise FullExecutionBenchmarkError(
                        f"{task.task_id} " "缺少 Prepared " "ResolvedChangeSurface"
                    )

                task_record = FullExecutionTaskRecord(
                    task_id=(task.task_id),
                    task_type=(task.task_type),
                    status="failed",
                    base_commit_sha=(current_sha),
                )

                analysis_root: Path | None = None

                try:
                    # ==================================
                    # Snapshot
                    #
                    # 当前 Task 必须基于最新成功
                    # Commit 创建只读分析快照。
                    # ==================================

                    analysis_root = await snapshot_manager.create(
                        task_id=(task.task_id),
                        commit_sha=(current_sha),
                    )

                    # ==================================
                    # Current Revision Context
                    #
                    # 每个 Task 都基于当前最新 Commit
                    # 重新计算：
                    #
                    # 1. File Index
                    # 2. Retrieval
                    # 3. Semantic Index
                    # 4. Dependency Graph
                    # 5. Final Selection
                    #
                    # 不能复用上一个 Repository
                    # Revision 的 Context。
                    # ==================================

                    context_selection = await self.context_selection_service.select(
                        repository_root=(analysis_root),
                        repository_revision=(current_sha),
                        task=task,
                        raw_requirement=(case.requirement),
                        requirement_spec=(requirement),
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
                        if (path not in retrieval_path_set)
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
                        update={"root_path": (str(analysis_root))},
                    )

                    # ==================================
                    # Base Current Context
                    #
                    # 这里先建立一份完全基于
                    # current_sha 的 Context。
                    #
                    # 注意：
                    #
                    # 此时还没有决定当前 Task
                    # 最终允许修改哪些文件。
                    # ==================================

                    base_bundle = self._base_bundle_for_task(
                        task=task,
                        requirement=(requirement),
                        technical_spec=(technical_spec),
                        repository=(current_repository),
                        current_root=(analysis_root),
                        selection=(selection),
                        engineering_rules=(engineering_rules),
                    )

                    # ==================================
                    # Fresh Change Surface
                    #
                    # 关键修复：
                    #
                    # 不再使用 Full Execution
                    # 开始前计算的旧 Surface。
                    #
                    # 当前 Task 必须结合：
                    #
                    # - 当前 Repository Revision
                    # - 当前 Retrieval
                    # - 当前代码 Context
                    # - 上游 Investigation
                    # - 本次 Run 已完成的
                    #   Execution Artifacts
                    #
                    # 重新判断：
                    #
                    # 当前还需要修改什么？
                    # ==================================

                    surface = await run_coordinator.resolve_change_surface(
                        bundle=(base_bundle),
                        selection=(selection),
                        task_plan=(task_plan),
                    )

                    task_record = task_record.model_copy(
                        deep=True,
                        update={
                            "execution_disposition": (surface.execution_disposition),
                        },
                    )

                    # ==================================
                    # Fresh No-op
                    #
                    # 原计划可能要求修改，
                    # 但前序 Task 已经完成了工作。
                    #
                    # 当前最新代码确认无需继续修改，
                    # 当前 Task 动态变成 no-op。
                    # ==================================

                    if surface.execution_disposition == "no_change_required":
                        task_record = task_record.model_copy(
                            deep=True,
                            update={
                                "status": ("skipped_no_change"),
                                "error": None,
                            },
                        )

                        task_records.append(task_record)

                        record = record.model_copy(
                            deep=True,
                            update={"tasks": (task_records)},
                        )

                        self.store.save(record)

                        continue

                    # ==================================
                    # Fresh Insufficient Evidence
                    #
                    # 只有真正查看过当前最新
                    # Repository State 后，
                    # 仍然无法确定安全修改范围，
                    # 才允许阻断。
                    # ==================================

                    if surface.execution_disposition == "insufficient_evidence":
                        unresolved_detail = (
                            " | ".join(surface.unresolved)
                            if surface.unresolved
                            else ("未提供具体 " "unresolved detail")
                        )

                        raise (
                            FullExecutionBenchmarkError(
                                "InsufficientEvidence: "
                                "基于当前最新 "
                                "Repository State "
                                "仍无法确定安全修改范围: " + unresolved_detail
                            )
                        )

                    # ==================================
                    # Task-type Write Boundary
                    #
                    # Change Surface：
                    #     当前哪些文件相关 /
                    #     哪些文件可能需要修改
                    #
                    # Execution Surface：
                    #     当前生命周期阶段
                    #     真正允许写哪些文件
                    #
                    # code_change:
                    #     production writable
                    #
                    # test:
                    #     tests writable
                    # ==================================

                    if task.task_type == "code_change":
                        execution_surface = production_surface_policy.apply(surface)

                    elif task.task_type == "test":
                        execution_surface = test_surface_policy.apply(surface)

                    else:
                        raise (
                            FullExecutionBenchmarkError(
                                "不支持的 " "Side Effect Task: " f"{task.task_type}"
                            )
                        )

                    # ==================================
                    # Current Surface 无可写目标
                    #
                    # 例如：
                    #
                    # Impact / Dependency 很多，
                    # 但当前生命周期阶段
                    # 没有真正允许修改的文件。
                    # ==================================

                    if not (execution_surface.requires_patch()):
                        task_record = task_record.model_copy(
                            deep=True,
                            update={
                                "status": ("skipped_no_production_change"),
                                "error": None,
                            },
                        )

                        task_records.append(task_record)

                        record = record.model_copy(
                            deep=True,
                            update={"tasks": (task_records)},
                        )

                        self.store.save(record)

                        continue

                    # ==================================
                    # Final Resolved Context
                    #
                    # 使用刚刚重新计算出来的
                    # Fresh Surface，
                    # 构造真正交给 Coding Agent
                    # 的代码上下文。
                    # ==================================

                    bundle = run_coordinator.build_resolved_context(
                        repository_root=(analysis_root),
                        requirement=(requirement),
                        technical_spec=(technical_spec),
                        repository=(current_repository),
                        index=(current_index),
                        bundle=(base_bundle),
                        selection=(selection),
                        surface=(execution_surface),
                        engineering_rules=(engineering_rules),
                        task_plan=(task_plan),
                    )

                    # ==================================
                    # Coding Analysis
                    # ==================================

                    # ==================================
                    # Coding Analysis
                    # ==================================

                    messages = coding_agent.prepare_messages(bundle)

                    analysis_messages = messages

                    analysis_attempts = 1

                    analysis = await coding_agent.analyze_prepared(
                        bundle=bundle,
                        messages=analysis_messages,
                    )

                    # ==================================
                    # Runtime Reconciliation
                    # ==================================

                    runtime_decision = runtime_execution_policy.decide(
                        analysis,
                        surface=execution_surface,
                    )

                    # ==================================
                    # Bounded Reconciliation Review
                    #
                    # 第一次分析：
                    #
                    #   no proposed changes
                    #       +
                    #   blocking evidence gap
                    #
                    # 不立刻降低 Safety Gate，
                    # 而是基于同一份最新 Revision Context
                    # 给 Agent 一次重新判断机会。
                    #
                    # 第二次仍 blocking：
                    #     Harness 正常阻断。
                    #
                    # 第二次确认无需修改：
                    #     skip_runtime_noop。
                    # ==================================

                    if (
                        runtime_decision.action == "block_insufficient_evidence"
                        and not analysis.proposed_changes
                    ):
                        analysis_messages = (
                            coding_agent.prepare_runtime_reconciliation_messages(
                                messages
                            )
                        )

                        analysis = await coding_agent.analyze_prepared(
                            bundle=bundle,
                            messages=(analysis_messages),
                        )

                        analysis_attempts += 1

                        runtime_decision = runtime_execution_policy.decide(
                            analysis,
                            surface=(execution_surface),
                        )

                    # ==================================
                    # Publish Final Analysis
                    #
                    # 注意：
                    # 发布的是最终真正被 Harness
                    # 用于 Runtime Decision 的分析结果。
                    # ==================================

                    context_hash = calculate_messages_hash(analysis_messages)

                    analysis_record = TaskAnalysisRecord(
                        task_id=(task.task_id),
                        status="succeeded",
                        attempts=(analysis_attempts),
                        result=analysis,
                        context_hash=(context_hash),
                    )

                    artifact, _ = analysis_store.publish(
                        bundle=bundle,
                        record=(analysis_record),
                        context_hash=(context_hash),
                    )

                    if runtime_decision.action == "skip_runtime_noop":
                        task_record = task_record.model_copy(
                            deep=True,
                            update={
                                "status": ("skipped_runtime_noop"),
                                "error": None,
                            },
                        )

                        task_records.append(task_record)

                        record = record.model_copy(
                            deep=True,
                            update={"tasks": (task_records)},
                        )

                        self.store.save(record)

                        continue

                    if runtime_decision.action == ("block_insufficient_evidence"):
                        raise (
                            FullExecutionBenchmarkError(
                                "RuntimeInsufficientEvidence: "
                                + (runtime_decision.reason)
                            )
                        )

                    # ==================================
                    # Patch Plan
                    # ==================================

                    snapshot = FileSnapshotBuilder(analysis_root).build(bundle)

                    patch_plan = PatchPlanner().plan(
                        artifact=(artifact),
                        snapshot=(snapshot),
                    )

                    if not patch_plan.targets:
                        raise (
                            FullExecutionBenchmarkError(
                                "Runtime invariant "
                                "violated: continue "
                                "decision produced an "
                                "empty PatchPlan"
                            )
                        )

                    plan_decision = run_coordinator.guard_patch_plan(
                        plan=(patch_plan),
                        surface=(execution_surface),
                    )

                    if not (plan_decision.allowed):
                        raise (
                            FullExecutionBenchmarkError(
                                "; ".join(plan_decision.reasons)
                            )
                        )

                    # ==================================
                    # Patch Proposal
                    # ==================================

                    preparation = ExecutionPreparer(
                        repository_root=(analysis_root),
                        artifact_store=(analysis_store),
                        message_provider=(coding_agent),
                    ).prepare(
                        task_id=(task.task_id),
                        artifact_id=(artifact.artifact_id),
                        bundle=bundle,
                    )

                    try:
                        proposal = await PatchGenerator(
                            repository_root=(analysis_root),
                            model=(text_model),
                        ).generate(
                            artifact=(artifact),
                            plan=(patch_plan),
                            preparation=(preparation),
                        )

                    except PatchGenerationNoop as noop:
                        # ==================================
                        # Late No-op Reconciliation
                        #
                        # Coding Analysis / PatchPlan
                        # 认为可能需要修改，
                        #
                        # 但 Patch Generation 在读取
                        # 当前真实文件后确认：
                        #
                        # 所有 Target 的最终内容
                        # 与当前内容完全一致。
                        #
                        # 此时：
                        #
                        # - 没有产生 Side Effect
                        # - 不创建 Workspace
                        # - 不产生 Commit
                        # - 不把它当成系统失败
                        #
                        # 对 test Task：
                        # 仍然保留对应测试类型，
                        # 让 Final Verification 执行。
                        # ==================================

                        noop_verification = None

                        if task.task_type == "test":
                            noop_verification = (
                                impact_verification_planner.build_for_test_files(
                                    base_command_ids=list(
                                        case.expectation.verification_command_ids
                                    ),
                                    test_files=list(noop.paths),
                                )
                            )

                        task_record = task_record.model_copy(
                            deep=True,
                            update={
                                "status": ("skipped_runtime_noop"),
                                "planned_verification": (noop_verification),
                                "error": None,
                            },
                        )

                        task_records.append(task_record)

                        record = record.model_copy(
                            deep=True,
                            update={"tasks": (task_records)},
                        )

                        self.store.save(record)

                        continue

                    proposal_decision = run_coordinator.guard_patch_proposal(
                        proposal=(proposal),
                        surface=(execution_surface),
                    )

                    if not (proposal_decision.allowed):
                        raise (
                            FullExecutionBenchmarkError(
                                "; ".join(proposal_decision.reasons)
                            )
                        )

                    planned_changed_files = sorted(
                        patch.relative_path for patch in proposal.patches
                    )

                    # ==================================
                    # Impact Analysis
                    # ==================================

                    planned_impact = impact_analyzer.analyze(
                        changed_files=(planned_changed_files),
                        graph=(context_selection.graph),
                        index=(current_index),
                        repository_revision=(current_sha),
                        # 使用原始 surface。
                        #
                        # Production Policy
                        # 会把 tests 转为只读；
                        # Test Policy 会改变
                        # production/test write
                        # boundary。
                        #
                        # Impact Analysis 需要
                        # 原始 Change Surface 中的
                        # 测试证据。
                        known_test_files=list(surface.tests),
                    )

                    # ==================================
                    # Impact-aware Verification Plan
                    # ==================================

                    planned_verification = impact_verification_planner.build(
                        base_command_ids=list(
                            case.expectation.verification_command_ids
                        ),
                        impact=(planned_impact),
                    )

                    task_record = task_record.model_copy(
                        deep=True,
                        update={
                            "planned_impact": (planned_impact),
                            "planned_verification": (planned_verification),
                        },
                    )

                    # ==================================
                    # Review / Approval
                    # ==================================

                    review = reviewer.prepare(proposal)

                    task_record = task_record.model_copy(
                        deep=True,
                        update={
                            "proposal_id": (review.proposal_id),
                            "proposed_files": (
                                sorted(
                                    patch.relative_path for patch in (proposal.patches)
                                )
                            ),
                        },
                    )

                    approval_store.record_decision(
                        review=review,
                        decision="approved",
                        reason=(
                            "benchmark "
                            "auto-approval: "
                            "schema + evidence + "
                            "change-surface "
                            "scope guard passed"
                        ),
                    )

                    # ==================================
                    # Workspace
                    # ==================================

                    workspace = await workspace_manager.create(
                        task_id=(task.task_id),
                        proposal_id=(review.proposal_id),
                        base_commit_sha=(current_sha),
                    )

                    task_record = task_record.model_copy(
                        deep=True,
                        update={"workspace_id": (workspace.workspace_id)},
                    )

                    self._assert_proposal_matches_execution_workspace(
                        proposal=proposal,
                        execution_root=Path(workspace.worktree_path),
                        analysis_root=(analysis_root),
                        commit_sha=(current_sha),
                    )

                    dependency_provisioner.provision(workspace.worktree_path)

                    # ==================================
                    # Staged Verification
                    #
                    # planned:
                    #   Impact Analysis 认为应该考虑
                    #
                    # scheduled:
                    #   当前生命周期阶段真正安排
                    #
                    # executed:
                    #   Runtime 真正执行
                    # ==================================

                    task_command_ids = task_verification_policy.for_task(
                        task=task,
                        base_command_ids=list(planned_verification.final_command_ids),
                    )

                    verification_plan = VerificationPlan(
                        proposal_id=(review.proposal_id),
                        task_id=(task.task_id),
                        command_ids=list(task_command_ids),
                    )

                    task_record = task_record.model_copy(
                        deep=True,
                        update={
                            "scheduled_verification_command_ids": (
                                list(task_command_ids)
                            ),
                        },
                    )

                    # ==================================
                    # Side Effect Execution
                    # ==================================

                    execution = await WorkspaceExecutionService(
                        runtime_root=(self.runtime_state_root),
                        approval_store=(approval_store),
                        journal_store=(journal_store),
                        verification_store=(verification_store),
                        completion_store=(completion_store),
                        verification_policy=(verification_policy),
                    ).execute(
                        workspace=(workspace),
                        proposal=(proposal),
                        verification_plan=(verification_plan),
                    )

                    verification_checks = []

                    if execution.verification.result is not None:
                        verification_checks = list(execution.verification.result.checks)

                    executed_command_ids = [
                        check.command_id
                        for check in verification_checks
                        if (check.command_id is not None)
                    ]

                    # ==================================
                    # Commit
                    # ==================================

                    commit = await commit_service.commit(
                        workspace=(workspace),
                        message=(
                            f"benchmark("
                            f"{case.case_id}"
                            f"): "
                            f"{task.task_id} "
                            f"{task.title}"
                        ),
                    )

                    # ==================================
                    # Execution Artifact
                    # ==================================

                    execution_artifact = TaskExecutionArtifactBuilder.build(
                        journal=(execution.journal),
                        completion=(execution.completion),
                        verification=(execution.verification),
                        base_revision=(current_sha),
                        result_revision=(commit.commit_sha),
                    )

                    run_coordinator.record_execution(
                        task=task,
                        artifact=(execution_artifact),
                    )

                    # ==================================
                    # Persist Task Execution Facts
                    # ==================================

                    task_record = task_record.model_copy(
                        deep=True,
                        update={
                            "status": ("completed"),
                            "workspace_id": (workspace.workspace_id),
                            "proposal_id": (review.proposal_id),
                            "proposed_files": (
                                sorted(
                                    patch.relative_path for patch in (proposal.patches)
                                )
                            ),
                            "scheduled_verification_command_ids": (
                                list(verification_plan.command_ids)
                            ),
                            "executed_verification_command_ids": (executed_command_ids),
                            "verification_status": (execution.verification.status),
                            "verification_checks": (verification_checks),
                            "completion_status": (execution.completion.status),
                            "commit_sha": (commit.commit_sha),
                            "error": None,
                        },
                    )

                    current_root = Path(workspace.worktree_path).resolve(strict=True)

                    current_sha = commit.commit_sha

                except Exception as exc:
                    error_message = f"{type(exc).__name__}: " f"{exc}"

                    verification_status = task_record.verification_status

                    verification_checks = list(task_record.verification_checks)

                    executed_verification_command_ids = list(
                        task_record.executed_verification_command_ids
                    )

                    # WorkspaceExecutionService
                    # 在 Verification 失败后只抛出
                    # compact gate error。
                    #
                    # VerificationStore 中保存了
                    # 真正的命令执行事实，
                    # 这里把它重新写回 Task Record。
                    if task_record.proposal_id and (
                        verification_store.exists(task_record.proposal_id)
                    ):
                        verification_record = verification_store.load(
                            task_record.proposal_id
                        )

                        verification_status = verification_record.status

                        if verification_record.result is not None:
                            verification_checks = list(
                                verification_record.result.checks
                            )

                            executed_verification_command_ids = [
                                check.command_id
                                for check in verification_checks
                                if (check.command_id is not None)
                            ]

                            failed_checks = [
                                check
                                for check in verification_checks
                                if (check.status == "failed")
                            ]

                            if failed_checks:
                                details = "\n\n".join(
                                    (
                                        f"[{check.name}] "
                                        f"{check.status}\n"
                                        f"{check.details}"
                                    )
                                    for check in failed_checks
                                )

                                error_message += (
                                    "\nVerification " "failed checks:\n" + details
                                )

                        elif verification_record.error:
                            error_message += "\nVerification " "workflow error: " + (
                                verification_record.error
                            )

                    task_record = task_record.model_copy(
                        deep=True,
                        update={
                            "verification_status": (verification_status),
                            "verification_checks": (verification_checks),
                            "executed_verification_command_ids": (
                                executed_verification_command_ids
                            ),
                            "error": (error_message),
                        },
                    )

                    task_records.append(task_record)

                    record = record.model_copy(
                        deep=True,
                        update={"tasks": (task_records)},
                    )

                    self.store.save(record)

                    raise

                finally:
                    if analysis_root is not None:
                        await snapshot_manager.remove(
                            analysis_root,
                            allow_missing=True,
                        )

                task_records.append(task_record)

                record = record.model_copy(
                    deep=True,
                    update={"tasks": (task_records)},
                )

                self.store.save(record)

            # ==================================
            # Hidden Acceptance
            # ==================================

            hidden = await HiddenAcceptanceRunner(
                fixture_root=(self.fe_agent_root / "benchmark" / "acceptance")
            ).run(
                case_id=(case.case_id),
                repository_root=(current_root),
            )

            # ==================================
            # Actual Change Set
            # ==================================

            actual_files = await GitDiffCollector().changed_files(
                repository_root=(self.benchmark_repository),
                base_commit_sha=(case.base_commit_sha),
                head_commit_sha=(current_sha),
            )

            # ==================================
            # Evaluation
            # ==================================

            evaluation = FullExecutionEvaluator().evaluate(
                case=case,
                actual_change_files=(actual_files),
                task_records=(task_records),
                hidden_acceptance=(hidden),
            )

            completed = record.model_copy(
                deep=True,
                update={
                    "status": ("completed"),
                    "finished_at": (datetime.now(timezone.utc)),
                    "final_commit_sha": (current_sha),
                    "final_workspace_root": (str(current_root)),
                    "tasks": (task_records),
                    "hidden_acceptance": (hidden),
                    "evaluation": (evaluation),
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
                    "finished_at": (datetime.now(timezone.utc)),
                    "final_commit_sha": (current_sha),
                    "final_workspace_root": (str(current_root)),
                    "tasks": (task_records),
                    "error": (f"{type(exc).__name__}: " f"{exc}"),
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
        """
        在 Apply 前确保：

            analysis snapshot
            execution workspace

        对 Proposal 目标文件拥有完全一致的
        base 内容。

        防止 Proposal 基于旧 Revision
        被错误应用到新 Workspace。
        """

        for patch in proposal.patches:
            path = execution_root / patch.relative_path

            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()

            except OSError as exc:
                raise (
                    FullExecutionBenchmarkError(
                        "Execution workspace "
                        "无法读取 Proposal 目标: "
                        f"{patch.relative_path}"
                    )
                ) from exc

            if actual != patch.expected_sha256:
                raise (
                    FullExecutionBenchmarkError(
                        "Proposal base 与 "
                        "execution workspace "
                        "不一致: "
                        f"path="
                        f"{patch.relative_path}, "
                        f"commit="
                        f"{commit_sha}, "
                        f"expected="
                        f"{patch.expected_sha256}, "
                        f"actual="
                        f"{actual}, "
                        f"analysis_root="
                        f"{analysis_root}, "
                        f"execution_root="
                        f"{execution_root}"
                    )
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
        # Import here to keep benchmark
        # module dependencies one-directional.
        from agent_core.context.builder import (
            ProjectContextBuilder,
        )

        return ProjectContextBuilder().build(
            repository_root=(current_root),
            requirement=(requirement),
            technical_spec=(technical_spec),
            task=task,
            repository=(repository),
            selection=(selection),
            engineering_rules=(engineering_rules),
            max_files=8,
            max_total_chars=40_000,
            max_chars_per_file=10_000,
        )
