from agent_core.git.handoff_model import (
    GitHandoffRecord,
)
from agent_core.git.pull_request_model import (
    PullRequestRequest,
)
from agent_core.git.push_model import (
    GitPushResult,
)


class PullRequestPreparationError(ValueError):
    pass


class PullRequestPreparationService:

    def prepare(
        self,
        *,
        handoff: GitHandoffRecord,
        push: GitPushResult,
        title: str,
    ) -> PullRequestRequest:

        # ----------------------------------
        # 1. Identity 校验
        # ----------------------------------

        if push.handoff_id != handoff.handoff_id:
            raise PullRequestPreparationError("Push Result " "不属于当前 Handoff")

        if push.task_id != handoff.task_id:
            raise PullRequestPreparationError("Push Result " "Task ID 不一致")

        if push.proposal_id != handoff.proposal_id:
            raise PullRequestPreparationError("Push Result " "Proposal ID 不一致")

        if push.branch_name != handoff.agent_branch:
            raise PullRequestPreparationError("Push Branch " "与 Handoff 不一致")

        if push.commit_sha != handoff.agent_commit_sha:
            raise PullRequestPreparationError("Push Commit " "与 Handoff Commit 不一致")

        # ----------------------------------
        # 2. Handoff 必须可以进入 PR
        # ----------------------------------

        if handoff.integration_status not in {
            "up_to_date",
            "clean",
        }:
            raise PullRequestPreparationError("当前 Handoff " "不允许创建 Pull Request")

        # ----------------------------------
        # 3. Title
        # ----------------------------------

        title = title.strip()

        if not title:
            raise PullRequestPreparationError("Pull Request title " "不能为空")

        # ----------------------------------
        # 4. 生成确定性的 PR Body
        # ----------------------------------

        body = self._build_body(
            handoff=handoff,
            push=push,
        )

        return PullRequestRequest(
            handoff_id=(handoff.handoff_id),
            task_id=(handoff.task_id),
            proposal_id=(handoff.proposal_id),
            remote_name=(push.remote_name),
            head_branch=(handoff.agent_branch),
            base_branch=(handoff.target_branch),
            commit_sha=(handoff.agent_commit_sha),
            title=title,
            body=body,
        )

    def _build_body(
        self,
        *,
        handoff: GitHandoffRecord,
        push: GitPushResult,
    ) -> str:

        file_lines = "\n".join(
            (
                f"- `{item.relative_path}`"
                f"\n"
                f"  - before: "
                f"`{item.before_sha256}`"
                f"\n"
                f"  - after: "
                f"`{item.after_sha256}`"
            )
            for item in handoff.files
        )

        return (
            "## Agent Change Summary\n\n"
            f"- Task: `{handoff.task_id}`\n"
            f"- Proposal: `{handoff.proposal_id}`\n"
            f"- Handoff: `{handoff.handoff_id}`\n"
            f"- Commit: `{handoff.agent_commit_sha}`\n"
            f"- Source Branch: "
            f"`{handoff.agent_branch}`\n"
            f"- Target Branch: "
            f"`{handoff.target_branch}`\n"
            f"- Integration: "
            f"`{handoff.integration_status}`\n"
            f"- Remote: `{push.remote_name}`\n\n"
            "## Changed Files\n\n"
            f"{file_lines}\n\n"
            "## Verification\n\n"
            "- Local Agent verification completed "
            "successfully before commit.\n"
            "- Git ChangeSet was validated "
            "against the approved Proposal.\n"
            "- Integration safety check "
            "completed without conflict.\n"
        )
