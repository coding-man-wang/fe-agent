import hashlib
from typing import Protocol

from agent_core.coding.approval import (
    ApprovalError,
)
from agent_core.coding.approval_store import (
    ApprovalStore,
)
from agent_core.coding.verification_plan import (
    VerificationPlan,
)
from agent_core.git.follow_up_task_model import (
    FollowUpTaskArtifact,
)
from agent_core.git.merge_readiness import (
    MergeReadinessGate,
)
from agent_core.git.pull_request_model import (
    PullRequestResult,
)
from agent_core.git.revision_push_model import (
    RevisionPushResult,
)
from agent_core.git.revision_task_model import (
    ExecutedRevisionTask,
    PreparedRevisionTask,
)
from agent_core.git.revision_workflow_state import (
    DurableRevisionTaskState,
    DurableRevisionWorkflowState,
)
from agent_core.git.revision_workflow_store import (
    DurableRevisionWorkflowStore,
)
from agent_core.git.ci_provider import (
    PullRequestCIProvider,
)
from agent_core.git.review_provider import (
    PullRequestReviewProvider,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


class RevisionTaskPreparerProtocol(Protocol):

    async def prepare(
        self,
        *,
        task: TaskDefinition,
        base_commit_sha: str,
    ) -> PreparedRevisionTask: ...


class ApprovedRevisionExecutorProtocol(Protocol):

    async def execute(
        self,
        *,
        task: TaskDefinition,
        prepared: PreparedRevisionTask,
        pull_request: PullRequestResult,
        verification_plan: VerificationPlan,
    ): ...


class RevisionPushProtocol(Protocol):

    async def push(
        self,
        *,
        prepared: PreparedRevisionTask,
        execution,
        pull_request: PullRequestResult,
        remote_name: str = "origin",
    ) -> RevisionPushResult: ...


class VerificationPlanProvider(Protocol):

    def create(
        self,
        *,
        task: TaskDefinition,
        prepared: PreparedRevisionTask,
    ) -> VerificationPlan: ...


class DurableRevisionError(RuntimeError):
    pass


class DurablePRRevisionOrchestrator:

    def __init__(
        self,
        *,
        store: DurableRevisionWorkflowStore,
        preparer: RevisionTaskPreparerProtocol,
        executor: ApprovedRevisionExecutorProtocol,
        push_service: RevisionPushProtocol,
        approval_store: ApprovalStore,
        verification_plan_provider: VerificationPlanProvider,
        ci_provider: PullRequestCIProvider,
        review_provider: PullRequestReviewProvider,
        merge_gate: MergeReadinessGate | None = None,
    ) -> None:

        self.store = store

        self.preparer = preparer
        self.executor = executor
        self.push_service = push_service

        self.approval_store = approval_store

        self.verification_plan_provider = verification_plan_provider

        self.ci_provider = ci_provider
        self.review_provider = review_provider

        self.merge_gate = merge_gate or MergeReadinessGate()

    # ==================================
    # Start
    # ==================================

    def start(
        self,
        *,
        follow_up: FollowUpTaskArtifact,
        pull_request: PullRequestResult,
    ) -> DurableRevisionWorkflowState:

        if follow_up.commit_sha != pull_request.commit_sha:

            raise DurableRevisionError("Follow-up Artifact " "与当前 PR Head 不一致")

        workflow_id = self._calculate_workflow_id(
            follow_up=follow_up,
            pull_request=pull_request,
        )

        tasks: list[DurableRevisionTaskState] = []

        base_commit = pull_request.commit_sha

        for task in follow_up.tasks:

            tasks.append(
                DurableRevisionTaskState(
                    task_id=task.task_id,
                    stage="pending",
                    base_commit_sha=(base_commit),
                )
            )

            # 注意：
            #
            # 后一个 Task 的真实 base，
            # 只有前一个执行完成后才知道。
            #
            # 初始这里只是占位。
            base_commit = ""

        state = DurableRevisionWorkflowState(
            workflow_id=(workflow_id),
            follow_up_artifact_id=(follow_up.artifact_id),
            repository=(pull_request.repository),
            pull_request_number=(pull_request.number),
            pull_request=(pull_request),
            status="preparing",
            current_task_index=0,
            tasks=tasks,
        )

        self.store.save(state)

        return state

    # ==================================
    # Prepare current task
    # ==================================

    async def prepare_current(
        self,
        *,
        workflow_id: str,
        follow_up: FollowUpTaskArtifact,
    ) -> DurableRevisionWorkflowState:

        state = self.store.load(workflow_id)

        self._require_status(
            state,
            {"preparing"},
        )

        task = self._current_task_definition(
            state=state,
            follow_up=follow_up,
        )

        task_state = state.tasks[state.current_task_index]

        if state.current_task_index == 0:
            base_commit = state.pull_request.commit_sha

        else:

            previous = state.tasks[state.current_task_index - 1]

            if previous.stage != "succeeded" or previous.commit_sha is None:
                raise DurableRevisionError("前一个 Revision Task " "尚未成功")

            base_commit = previous.commit_sha

        try:

            prepared = await self.preparer.prepare(
                task=task,
                base_commit_sha=(base_commit),
            )

            if prepared.task_id != task.task_id:
                raise DurableRevisionError("Preparer 返回错误 Task")

            if prepared.base_commit_sha != base_commit:
                raise DurableRevisionError("Prepared Revision " "Base Commit 错误")

            updated_task = task_state.model_copy(
                update={
                    "stage": "awaiting_approval",
                    "base_commit_sha": base_commit,
                    "prepared": prepared,
                    "error": None,
                }
            )

            state.tasks[state.current_task_index] = updated_task

            state = state.model_copy(
                update={
                    "status": "awaiting_approval",
                    "error": None,
                }
            )

            self.store.save(state)

            return state

        except Exception as exc:

            return self._fail(
                state=state,
                task_index=(state.current_task_index),
                exc=exc,
            )

    # ==================================
    # Execute approved proposal
    # ==================================

    async def execute_current(
        self,
        *,
        workflow_id: str,
        follow_up: FollowUpTaskArtifact,
        remote_name: str = "origin",
    ) -> DurableRevisionWorkflowState:

        state = self.store.load(workflow_id)

        self._require_status(
            state,
            {"awaiting_approval"},
        )

        task = self._current_task_definition(
            state=state,
            follow_up=follow_up,
        )

        task_state = state.tasks[state.current_task_index]

        prepared = task_state.prepared

        if prepared is None:

            raise DurableRevisionError("当前 Revision Task " "缺少 Prepared Proposal")

        # -----------------------------
        # Human approval boundary
        # -----------------------------

        try:

            receipt = self.approval_store.load(prepared.proposal_id)

        except ApprovalError:

            # 尚未审批不是失败。
            #
            # 状态保持 awaiting_approval。
            return state

        if receipt.decision == "rejected":

            updated_task = task_state.model_copy(
                update={
                    "stage": "failed",
                    "error": ("Proposal 被拒绝: " f"{receipt.reason}"),
                }
            )

            state.tasks[state.current_task_index] = updated_task

            state = state.model_copy(
                update={
                    "status": "failed",
                    "error": ("Revision Proposal " "被人工拒绝"),
                }
            )

            self.store.save(state)

            return state

        if receipt.task_id != task.task_id:

            raise DurableRevisionError("Approval Task ID 不一致")

        # -----------------------------
        # Side effect 前先 checkpoint
        # -----------------------------

        state = state.model_copy(update={"status": "executing"})

        self.store.save(state)

        try:

            verification_plan = self.verification_plan_provider.create(
                task=task,
                prepared=prepared,
            )

            execution = await self.executor.execute(
                task=task,
                prepared=prepared,
                pull_request=(state.pull_request),
                verification_plan=(verification_plan),
            )

            # -----------------------------
            # Push exact commit
            # -----------------------------

            push = await self.push_service.push(
                prepared=prepared,
                execution=execution,
                pull_request=(state.pull_request),
                remote_name=(remote_name),
            )

            new_commit = push.pushed_commit_sha

            updated_task = task_state.model_copy(
                update={
                    "stage": "succeeded",
                    "commit_sha": new_commit,
                    "error": None,
                }
            )

            state.tasks[state.current_task_index] = updated_task

            updated_pr = state.pull_request.model_copy(
                update={"commit_sha": new_commit}
            )

            next_index = state.current_task_index + 1

            # -----------------------------
            # 还有下一个 Revision Task
            # -----------------------------

            if next_index < len(state.tasks):

                state = state.model_copy(
                    update={
                        "pull_request": updated_pr,
                        "current_task_index": next_index,
                        "status": "preparing",
                        "latest_ci": None,
                        "latest_review": None,
                        "error": None,
                    }
                )

            else:

                # 所有返修 Task 已执行，
                # 下一阶段进入远端 CI。
                state = state.model_copy(
                    update={
                        "pull_request": updated_pr,
                        "status": "waiting_ci",
                        "latest_ci": None,
                        "latest_review": None,
                        "error": None,
                    }
                )

            self.store.save(state)

            return state

        except Exception as exc:

            return self._fail(
                state=state,
                task_index=(state.current_task_index),
                exc=exc,
            )

    # ==================================
    # Remote CI / Review refresh
    # ==================================

    async def refresh_remote(
        self,
        *,
        workflow_id: str,
    ) -> DurableRevisionWorkflowState:

        state = self.store.load(workflow_id)

        self._require_status(
            state,
            {
                "waiting_ci",
                "waiting_review",
            },
        )

        try:

            # -----------------------------
            # CI 一次快照
            # -----------------------------

            ci = await self.ci_provider.get_status(pull_request=(state.pull_request))

            if ci.commit_sha != state.pull_request.commit_sha:
                raise DurableRevisionError("CI 状态不属于 " "当前 PR Head Commit")

            if ci.status in {
                "pending",
                "no_checks",
            }:

                state = state.model_copy(
                    update={
                        "status": "waiting_ci",
                        "latest_ci": ci,
                    }
                )

                self.store.save(state)

                return state

            if ci.status == "failed":

                state = state.model_copy(
                    update={
                        "status": "failed",
                        "latest_ci": ci,
                        "error": "Remote CI failed",
                    }
                )

                self.store.save(state)

                return state

            # -----------------------------
            # CI 成功后才查 Review
            # -----------------------------

            review = await self.review_provider.get_status(
                pull_request=(state.pull_request)
            )

            if review.commit_sha != state.pull_request.commit_sha:

                raise DurableRevisionError("Review 状态不属于 " "当前 PR Head Commit")

            if review.status == "changes_requested":

                state = state.model_copy(
                    update={
                        "status": "changes_requested",
                        "latest_ci": ci,
                        "latest_review": review,
                        "error": None,
                    }
                )

                self.store.save(state)

                return state

            if review.status == "pending":

                state = state.model_copy(
                    update={
                        "status": "waiting_review",
                        "latest_ci": ci,
                        "latest_review": review,
                        "error": None,
                    }
                )

                self.store.save(state)

                return state

            # -----------------------------
            # Merge Readiness
            # -----------------------------

            readiness = self.merge_gate.evaluate(
                ci=self._to_ci_tracking(ci),
                review=review,
            )

            if readiness.status == "ready_for_human_merge":

                state = state.model_copy(
                    update={
                        "status": "ready_for_human_merge",
                        "latest_ci": ci,
                        "latest_review": review,
                        "error": None,
                    }
                )

            else:

                state = state.model_copy(
                    update={
                        "status": "waiting_review",
                        "latest_ci": ci,
                        "latest_review": review,
                    }
                )

            self.store.save(state)

            return state

        except Exception as exc:

            return self._fail(
                state=state,
                task_index=None,
                exc=exc,
            )

    # ==================================
    # Helpers
    # ==================================

    def _current_task_definition(
        self,
        *,
        state: DurableRevisionWorkflowState,
        follow_up: FollowUpTaskArtifact,
    ) -> TaskDefinition:

        if follow_up.artifact_id != state.follow_up_artifact_id:
            raise DurableRevisionError("Follow-up Artifact " "与 Workflow 不一致")

        if state.current_task_index >= len(follow_up.tasks):
            raise DurableRevisionError("current_task_index 越界")

        task = follow_up.tasks[state.current_task_index]

        state_task = state.tasks[state.current_task_index]

        if task.task_id != state_task.task_id:
            raise DurableRevisionError("Workflow Task " "与 Follow-up Task 不一致")

        return task

    def _require_status(
        self,
        state: DurableRevisionWorkflowState,
        allowed: set[str],
    ) -> None:

        if state.status not in allowed:

            raise DurableRevisionError(
                "当前 Workflow 状态 " f"{state.status} " "不允许执行此操作"
            )

    def _fail(
        self,
        *,
        state: DurableRevisionWorkflowState,
        task_index: int | None,
        exc: Exception,
    ) -> DurableRevisionWorkflowState:

        message = f"{type(exc).__name__}: " f"{exc}"

        if task_index is not None:

            task = state.tasks[task_index]

            state.tasks[task_index] = task.model_copy(
                update={
                    "stage": "failed",
                    "error": message,
                }
            )

        state = state.model_copy(
            update={
                "status": "failed",
                "error": message,
            }
        )

        self.store.save(state)

        return state

    @staticmethod
    def _calculate_workflow_id(
        *,
        follow_up: FollowUpTaskArtifact,
        pull_request: PullRequestResult,
    ) -> str:

        raw = (
            f"{follow_up.artifact_id}:"
            f"{pull_request.repository}:"
            f"{pull_request.number}:"
            f"{pull_request.commit_sha}"
        )

        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _to_ci_tracking(
        ci,
    ):
        from agent_core.git.ci_tracking_model import (
            CITrackingResult,
        )

        outcome = "succeeded" if ci.status == "succeeded" else ci.status

        return CITrackingResult(
            outcome=outcome,
            attempts=1,
            final_status=ci,
        )

    def recover_interrupted(
        self,
        *,
        workflow_id: str,
    ) -> DurableRevisionWorkflowState:

        state = self.store.load(workflow_id)

        if state.status != "executing":
            return state

        state = state.model_copy(
            update={
                "status": "failed",
                "error": (
                    "InterruptedExecution: "
                    "Revision 在 Side Effect "
                    "执行期间中断，需要显式对账恢复"
                ),
            }
        )

        self.store.save(state)

        return state
