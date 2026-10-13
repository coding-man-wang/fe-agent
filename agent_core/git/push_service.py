from pathlib import Path

from agent_core.git.client import (
    GitClient,
)
from agent_core.git.cli_client import (
    CliGitClient,
)
from agent_core.git.handoff_model import (
    GitHandoffRecord,
)
from agent_core.git.push_model import (
    GitPushResult,
)


class GitPushError(RuntimeError):
    pass


class AgentGitPushService:

    def __init__(
        self,
        *,
        git_client: GitClient | None = None,
    ) -> None:

        self.git_client = git_client or CliGitClient()

    async def push(
        self,
        *,
        handoff: GitHandoffRecord,
        remote_name: str = "origin",
    ) -> GitPushResult:

        repository_root = Path(handoff.source_repository_root).resolve()

        if not repository_root.is_dir():
            raise GitPushError("Source Repository 不存在")

        remote_name = remote_name.strip()

        if not remote_name:
            raise GitPushError("remote_name 不能为空")

        # --------------------------
        # 1. 确认 Remote 存在
        # --------------------------

        remote_result = await self.git_client.run(
            cwd=repository_root,
            args=[
                "remote",
            ],
        )

        remotes = {
            line.strip() for line in (remote_result.stdout.splitlines()) if line.strip()
        }

        if remote_name not in remotes:
            raise GitPushError(f"Git Remote 不存在: " f"{remote_name}")

        # --------------------------
        # 2. 确认 Agent Branch
        #    当前仍指向 Handoff Commit
        # --------------------------

        branch_result = await self.git_client.run(
            cwd=repository_root,
            args=[
                "rev-parse",
                handoff.agent_branch,
            ],
        )

        branch_head = branch_result.stdout.strip()

        if branch_head != handoff.agent_commit_sha:
            raise GitPushError("Agent Branch HEAD " "已经发生变化")

        # --------------------------
        # 3. 确认 Commit 确实存在
        # --------------------------

        await self.git_client.run(
            cwd=repository_root,
            args=[
                "cat-file",
                "-e",
                (handoff.agent_commit_sha + "^{commit}"),
            ],
        )

        # --------------------------
        # 4. 检查远端同名分支
        # --------------------------

        remote_branch = f"refs/heads/" f"{handoff.agent_branch}"

        remote_check = await self.git_client.run(
            cwd=repository_root,
            args=[
                "ls-remote",
                "--heads",
                remote_name,
                remote_branch,
            ],
        )

        remote_sha = None

        line = remote_check.stdout.strip()

        if line:
            parts = line.split()

            if len(parts) < 2:
                raise GitPushError("无法解析远端分支状态")

            remote_sha = parts[0]

        # --------------------------
        # 5. 远端已经是同一 Commit
        #    直接认为幂等成功
        # --------------------------

        if remote_sha == handoff.agent_commit_sha:
            return GitPushResult(
                handoff_id=(handoff.handoff_id),
                task_id=(handoff.task_id),
                proposal_id=(handoff.proposal_id),
                remote_name=remote_name,
                branch_name=(handoff.agent_branch),
                commit_sha=(handoff.agent_commit_sha),
                remote_ref=(
                    f"refs/remotes/" f"{remote_name}/" f"{handoff.agent_branch}"
                ),
            )

        # --------------------------
        # 6. 远端已存在不同 Commit
        #    v0.1 直接拒绝
        # --------------------------

        if remote_sha is not None:
            raise GitPushError("远端 Agent Branch " "已经存在不同 Commit，" "拒绝覆盖")

        # --------------------------
        # 7. Push
        #
        # 明确 source:destination，
        # 不依赖当前 checkout branch。
        # --------------------------

        refspec = (
            f"{handoff.agent_commit_sha}" f":refs/heads/" f"{handoff.agent_branch}"
        )

        await self.git_client.run(
            cwd=repository_root,
            args=[
                "push",
                remote_name,
                refspec,
            ],
        )

        # --------------------------
        # 8. Push 后重新确认远端
        # --------------------------

        verify = await self.git_client.run(
            cwd=repository_root,
            args=[
                "ls-remote",
                "--heads",
                remote_name,
                remote_branch,
            ],
        )

        verify_line = verify.stdout.strip()

        if not verify_line:
            raise GitPushError("Push 后未找到远端分支")

        verify_sha = verify_line.split()[0]

        if verify_sha != handoff.agent_commit_sha:
            raise GitPushError("远端 Commit " "与 Handoff Commit 不一致")

        return GitPushResult(
            handoff_id=(handoff.handoff_id),
            task_id=(handoff.task_id),
            proposal_id=(handoff.proposal_id),
            remote_name=remote_name,
            branch_name=(handoff.agent_branch),
            commit_sha=(handoff.agent_commit_sha),
            remote_ref=(f"refs/remotes/" f"{remote_name}/" f"{handoff.agent_branch}"),
        )
