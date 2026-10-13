import hashlib
from typing import Protocol

from agent_core.git.follow_up_task_model import (
    FollowUpTaskArtifact,
)
from agent_core.git.pull_request_model import (
    PullRequestResult,
)
from agent_core.git.revision_model import (
    RevisionTaskResult,
    RevisionWorkflowResult,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


class RevisionWorkflowError(ValueError):
    pass


class RevisionTaskExecutionResult(Protocol):

    task_id: str
    base_commit_sha: str
    commit_sha: str


class RevisionTaskExecutor(Protocol):

    async def execute(
        self,
        *,
        task: TaskDefinition,
        pull_request: PullRequestResult,
        base_commit_sha: str,
    ) -> RevisionTaskExecutionResult:
        """
        必须：

        1. 基于 base_commit_sha 建立新的代码快照；
        2. 在这个快照上完成 Retrieval / Context；
        3. 执行 Analysis / Patch / Approval；
        4. Apply / Verification / Completion；
        5. Commit；
        6. Push 到 pull_request.head_branch；
        7. 返回真正产生的新 commit_sha。
        """
        ...


class RevisionWorkflow:

    def __init__(
        self,
        *,
        executor: RevisionTaskExecutor,
    ) -> None:

        self.executor = executor

    async def run(
        self,
        *,
        follow_up: FollowUpTaskArtifact,
        pull_request: PullRequestResult,
    ) -> RevisionWorkflowResult:

        self._validate_inputs(
            follow_up=follow_up,
            pull_request=pull_request,
        )

        revision_id = self._calculate_revision_id(
            follow_up=follow_up,
            pull_request=pull_request,
        )

        initial_commit_sha = pull_request.commit_sha

        current_commit_sha = initial_commit_sha

        results: list[RevisionTaskResult] = []

        for task in follow_up.tasks:

            base_commit_sha = current_commit_sha

            try:

                execution = await self.executor.execute(
                    task=task,
                    pull_request=(pull_request),
                    base_commit_sha=(base_commit_sha),
                )

                self._validate_execution(
                    task=task,
                    expected_base_commit_sha=(base_commit_sha),
                    execution=execution,
                )

                current_commit_sha = execution.commit_sha

                results.append(
                    RevisionTaskResult(
                        task_id=(task.task_id),
                        status="succeeded",
                        base_commit_sha=(base_commit_sha),
                        commit_sha=(current_commit_sha),
                        error=None,
                    )
                )

            except Exception as exc:

                results.append(
                    RevisionTaskResult(
                        task_id=(task.task_id),
                        status="failed",
                        base_commit_sha=(base_commit_sha),
                        commit_sha=None,
                        error=str(exc),
                    )
                )

                return RevisionWorkflowResult(
                    revision_id=(revision_id),
                    source_follow_up_artifact_id=(follow_up.artifact_id),
                    pull_request_number=(pull_request.number),
                    branch_name=(pull_request.head_branch),
                    initial_commit_sha=(initial_commit_sha),
                    final_commit_sha=(current_commit_sha),
                    status="failed",
                    tasks=results,
                    updated_pull_request=None,
                )

        updated_pull_request = pull_request.model_copy(
            update={"commit_sha": current_commit_sha}
        )

        return RevisionWorkflowResult(
            revision_id=revision_id,
            source_follow_up_artifact_id=(follow_up.artifact_id),
            pull_request_number=(pull_request.number),
            branch_name=(pull_request.head_branch),
            initial_commit_sha=(initial_commit_sha),
            final_commit_sha=(current_commit_sha),
            status="succeeded",
            tasks=results,
            updated_pull_request=(updated_pull_request),
        )

    def _validate_inputs(
        self,
        *,
        follow_up: FollowUpTaskArtifact,
        pull_request: PullRequestResult,
    ) -> None:

        if follow_up.commit_sha != pull_request.commit_sha:
            raise RevisionWorkflowError(
                "Follow-up Task 所基于的 " "Commit 与当前 PR Head 不一致"
            )

        if not pull_request.head_branch:
            raise RevisionWorkflowError("PR Head Branch 为空")

        task_ids = [task.task_id for task in follow_up.tasks]

        if len(task_ids) != len(set(task_ids)):
            raise RevisionWorkflowError("Follow-up Task " "存在重复 task_id")

        link_task_ids = {link.follow_up_task_id for link in follow_up.links}

        if set(task_ids) != link_task_ids:
            raise RevisionWorkflowError(
                "Follow-up Task " "与 Feedback Link " "范围不一致"
            )

    def _validate_execution(
        self,
        *,
        task: TaskDefinition,
        expected_base_commit_sha: str,
        execution: RevisionTaskExecutionResult,
    ) -> None:

        if execution.task_id != task.task_id:
            raise RevisionWorkflowError("Revision Executor " "返回了错误的 task_id")

        if execution.base_commit_sha != expected_base_commit_sha:
            raise RevisionWorkflowError("Revision Executor " "没有基于预期 Commit 执行")

        if not execution.commit_sha:
            raise RevisionWorkflowError("Revision Executor " "没有产生 Commit")

        if execution.commit_sha == expected_base_commit_sha:
            raise RevisionWorkflowError("Revision Executor " "没有产生新的 Commit")

    def _calculate_revision_id(
        self,
        *,
        follow_up: FollowUpTaskArtifact,
        pull_request: PullRequestResult,
    ) -> str:

        raw = (
            f"{follow_up.artifact_id}:"
            f"{pull_request.repository}:"
            f"{pull_request.number}:"
            f"{pull_request.head_branch}:"
            f"{pull_request.commit_sha}"
        )

        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
