from pathlib import Path
from typing import Protocol

from agent_core.coding.agent import (
    ReadOnlyCodingAgent,
)
from agent_core.coding.handoff import (
    AnalysisArtifact,
    AnalysisArtifactStore,
    calculate_messages_hash,
)
from agent_core.coding.orchestration_model import (
    TaskAnalysisRecord,
)
from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
)
from agent_core.coding.patch_generator import (
    PatchGenerator,
)
from agent_core.coding.patch_planner import (
    PatchPlanner,
)
from agent_core.coding.preparation import (
    ExecutionPreparer,
)
from agent_core.coding.snapshot import (
    FileSnapshotBuilder,
)
from agent_core.context.factory import (
    DefaultTaskContextFactory,
)
from agent_core.context.model import (
    TaskContextBundle,
)
from agent_core.git.revision_code_snapshot import (
    RevisionCodeSnapshot,
    RevisionCodeSnapshotFactory,
)
from agent_core.git.revision_task_model import (
    PreparedRevisionTask,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


class RevisionTaskPreparationError(ValueError):
    pass


class RevisionContextFactoryProvider(Protocol):

    def create(
        self,
        *,
        code_snapshot: RevisionCodeSnapshot,
    ) -> DefaultTaskContextFactory: ...


class RevisionPatchGeneratorProvider(Protocol):

    def create(
        self,
        *,
        repository_root: Path,
    ) -> PatchGenerator: ...


class RevisionTaskPreparer:

    def __init__(
        self,
        *,
        snapshot_factory: RevisionCodeSnapshotFactory,
        context_factory_provider: RevisionContextFactoryProvider,
        coding_agent: ReadOnlyCodingAgent,
        artifact_store: AnalysisArtifactStore,
        patch_generator_provider: RevisionPatchGeneratorProvider,
        patch_planner: PatchPlanner | None = None,
    ) -> None:

        self.snapshot_factory = snapshot_factory

        self.context_factory_provider = context_factory_provider

        self.coding_agent = coding_agent

        self.artifact_store = artifact_store

        self.patch_generator_provider = patch_generator_provider

        self.patch_planner = patch_planner or PatchPlanner()

    async def prepare(
        self,
        *,
        task: TaskDefinition,
        base_commit_sha: str,
    ) -> PreparedRevisionTask:

        # ----------------------------------
        # 1. 创建固定 Git Snapshot
        # ----------------------------------

        code_snapshot = await self.snapshot_factory.create(
            task=task,
            base_commit_sha=(base_commit_sha),
        )

        snapshot_root = Path(code_snapshot.snapshot.snapshot_path).resolve(strict=True)

        # ----------------------------------
        # 2. Context 必须基于同一 Snapshot
        # ----------------------------------

        context_factory = self.context_factory_provider.create(
            code_snapshot=(code_snapshot)
        )

        bundle = await context_factory.build(task)

        self._validate_bundle(
            task=task,
            bundle=bundle,
        )

        # ----------------------------------
        # 3. Read-only Analysis
        # ----------------------------------

        messages = self.coding_agent.prepare_messages(bundle)

        context_hash = calculate_messages_hash(messages)

        result = await self.coding_agent.analyze_prepared(
            bundle=bundle,
            messages=messages,
        )

        if result.task_id != task.task_id:
            raise (RevisionTaskPreparationError("Analysis Task ID 不一致"))

        # ----------------------------------
        # 4. 发布 Analysis Artifact
        # ----------------------------------

        record = TaskAnalysisRecord(
            task_id=task.task_id,
            status="succeeded",
            attempts=1,
            result=result,
        )

        artifact, _ = self.artifact_store.publish(
            bundle=bundle,
            record=record,
            context_hash=(context_hash),
        )

        # ----------------------------------
        # 5. 对 Snapshot 文件做 SHA 快照
        # ----------------------------------

        file_snapshot_builder = FileSnapshotBuilder(snapshot_root)

        file_snapshot = file_snapshot_builder.build(bundle)

        # ----------------------------------
        # 6. 生成 PatchPlan
        # ----------------------------------

        patch_plan = self.patch_planner.plan(
            artifact=artifact,
            snapshot=file_snapshot,
        )

        if not patch_plan.targets:
            raise (
                RevisionTaskPreparationError(
                    "Revision Analysis " "没有产生代码修改目标"
                )
            )

        # ----------------------------------
        # 7. Execution Preparation
        #
        # 仍然针对 Analysis Snapshot 做检查
        # ----------------------------------

        preparer = ExecutionPreparer(
            repository_root=(snapshot_root),
            artifact_store=(self.artifact_store),
            message_provider=(self.coding_agent),
        )

        preparation = preparer.prepare(
            task_id=task.task_id,
            artifact_id=(artifact.artifact_id),
            bundle=bundle,
        )

        # ----------------------------------
        # 8. 生成精确 Proposal
        # ----------------------------------

        patch_generator = self.patch_generator_provider.create(
            repository_root=(snapshot_root)
        )

        proposal = await patch_generator.generate(
            artifact=artifact,
            plan=patch_plan,
            preparation=preparation,
        )

        proposal_id = self._calculate_proposal_id(proposal)

        return PreparedRevisionTask(
            task_id=task.task_id,
            base_commit_sha=(base_commit_sha),
            snapshot_id=(code_snapshot.snapshot.snapshot_id),
            analysis_artifact_id=(artifact.artifact_id),
            proposal_id=proposal_id,
            proposal=proposal,
        )

    def _validate_bundle(
        self,
        *,
        task: TaskDefinition,
        bundle: TaskContextBundle,
    ) -> None:

        if bundle.task.task_id != task.task_id:
            raise (RevisionTaskPreparationError("Context Task ID 不一致"))

        if not bundle.code_context:
            raise (
                RevisionTaskPreparationError("Revision Task " "没有检索到代码上下文")
            )

    def _calculate_proposal_id(
        self,
        proposal: FilePatchProposal,
    ) -> str:

        # Proposal ID 的唯一计算规则
        # 仍然复用 PatchReviewer。
        from agent_core.coding.reviewer import (
            PatchReviewer,
        )

        review = PatchReviewer().prepare(proposal)

        return review.proposal_id
