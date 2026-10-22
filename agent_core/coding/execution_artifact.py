from agent_core.artifacts.execution_model import (
    ExecutedFileChange,
    ExecutionVerificationCheck,
    TaskExecutionArtifact,
)
from agent_core.coding.execution_journal_model import (
    PatchExecutionJournal,
)
from agent_core.coding.task_completion_model import (
    TaskCompletionRecord,
)
from agent_core.coding.verification_record_model import (
    VerificationRunRecord,
)


class ExecutionArtifactBuildError(ValueError):
    pass


class TaskExecutionArtifactBuilder:
    """
    把 Coding Runtime 的最终执行事实
    转换成 Harness 可以跨 Task 消费的
    TaskExecutionArtifact。

    这是一个 Adapter / Bridge。

    Adapter：
        把一种数据结构转换成另一种
        系统能够理解的数据结构。

    这里不是重新执行任何逻辑，
    只是转换已经验证过的事实。
    """

    @staticmethod
    def build(
        *,
        journal: PatchExecutionJournal,
        completion: TaskCompletionRecord,
        verification: VerificationRunRecord,
    ) -> TaskExecutionArtifact:

        # =================================
        # 1. Identity
        #
        # 三个 Runtime Record
        # 必须描述同一个 Task / Proposal。
        # =================================

        if not (journal.task_id == completion.task_id == verification.task_id):
            raise ExecutionArtifactBuildError(
                "Execution Runtime Record " "的 task_id 不一致"
            )

        if not (
            journal.proposal_id == completion.proposal_id == verification.proposal_id
        ):
            raise ExecutionArtifactBuildError(
                "Execution Runtime Record " "的 proposal_id 不一致"
            )

        # =================================
        # 2. Execution 必须已经完成
        # =================================

        if journal.status != "applied":
            raise ExecutionArtifactBuildError(
                "Patch 尚未完整 Apply，" "不能发布 Execution Artifact"
            )

        if any(item.status != "applied" for item in journal.files):
            raise ExecutionArtifactBuildError("仍存在未 applied 的文件")

        # =================================
        # 3. Completion 必须是真 completed
        # =================================

        if completion.status != "completed":
            raise ExecutionArtifactBuildError("Task 尚未 completed")

        if completion.execution_status != "applied":
            raise ExecutionArtifactBuildError(
                "Completion Record " "未确认 execution=applied"
            )

        if completion.verification_status != "passed":
            raise ExecutionArtifactBuildError(
                "Completion Record " "未确认 verification=passed"
            )

        # =================================
        # 4. Verification
        # =================================

        if verification.status != "succeeded":
            raise ExecutionArtifactBuildError("Verification Workflow " "尚未成功")

        if verification.result is None:
            raise ExecutionArtifactBuildError("Verification Result 缺失")

        if verification.result.status != "passed":
            raise ExecutionArtifactBuildError("Verification 没有通过")

        if verification.result.task_id != journal.task_id:
            raise ExecutionArtifactBuildError("Verification Result " "task_id 不一致")

        if verification.result.proposal_id != journal.proposal_id:
            raise ExecutionArtifactBuildError(
                "Verification Result " "proposal_id 不一致"
            )

        # =================================
        # 5. 防止同一个文件重复出现
        # =================================

        paths = [item.relative_path for item in journal.files]

        if len(paths) != len(set(paths)):
            raise ExecutionArtifactBuildError("Execution Journal " "包含重复文件")

        # =================================
        # 6. Verification Checks
        # =================================

        verification_checks: list[ExecutionVerificationCheck] = []

        for check in verification.result.checks:
            if check.status != "passed":
                raise ExecutionArtifactBuildError(
                    "Verification Result " "包含未通过检查"
                )

            verification_checks.append(
                ExecutionVerificationCheck(
                    name=check.name,
                    status="passed",
                    details=check.details,
                )
            )

        # =================================
        # 7. Publishable Artifact
        # =================================

        return TaskExecutionArtifact(
            task_id=journal.task_id,
            proposal_id=journal.proposal_id,
            changed_files=[
                ExecutedFileChange(
                    relative_path=(item.relative_path),
                    before_sha256=(item.expected_sha256),
                    after_sha256=(item.after_sha256),
                    status="applied",
                )
                for item in journal.files
            ],
            verification_checks=(verification_checks),
        )
