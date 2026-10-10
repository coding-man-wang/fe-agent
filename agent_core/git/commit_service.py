from pathlib import Path

from agent_core.git.client import (
    GitClient,
)
from agent_core.git.cli_client import (
    CliGitClient,
)
from agent_core.git.commit_model import (
    GitCommitResult,
)
from agent_core.git.errors import (
    GitWorkspaceConflictError,
)
from agent_core.git.repository import (
    GitRepositoryInspector,
)
from agent_core.git.workspace_model import (
    GitWorkspaceRecord,
)


class GitCommitError(RuntimeError):
    pass


class AgentGitCommitService:

    def __init__(
        self,
        *,
        git_client: GitClient | None = None,
        inspector: GitRepositoryInspector | None = None,
    ) -> None:

        self.git_client = git_client or CliGitClient()

        self.inspector = inspector or GitRepositoryInspector(git_client=self.git_client)

    async def commit(
        self,
        *,
        workspace: GitWorkspaceRecord,
        message: str,
    ) -> GitCommitResult:

        if workspace.status != "ready":
            raise GitWorkspaceConflictError("Workspace 不是 ready 状态")

        worktree = Path(workspace.worktree_path).resolve()

        if not worktree.is_dir():
            raise GitWorkspaceConflictError("Worktree 不存在")

        message = message.strip()

        if not message:
            raise GitCommitError("Commit message 不能为空")

        # 1. 再确认当前目录真的是这个 Workspace
        state = await self.inspector.inspect(worktree)

        if state.branch_name != workspace.branch_name:
            raise GitWorkspaceConflictError(
                "当前 Worktree Branch " "与 Workspace Record 不一致"
            )

        # 2. 必须真的有变化
        if state.working_tree.is_clean:
            raise GitCommitError("Worktree 没有可提交的修改")

        # 3. 记录 Commit 前 HEAD
        parent_sha = state.head_sha

        # 4. Stage 所有修改
        await self.git_client.run(
            cwd=worktree,
            args=[
                "add",
                "--all",
            ],
        )

        # 5. 防止 staged 内容为空
        diff_result = await self.git_client.run(
            cwd=worktree,
            args=[
                "diff",
                "--cached",
                "--quiet",
            ],
            allow_failure=True,
        )

        # git diff --quiet:
        # 0 = 没有差异
        # 1 = 有差异
        # >1 = 真正错误
        if diff_result.exit_code == 0:
            raise GitCommitError("没有 staged 修改可提交")

        if diff_result.exit_code != 1:
            raise GitCommitError("无法检查 staged diff")

        # 6. 真正 Commit
        await self.git_client.run(
            cwd=worktree,
            args=[
                "commit",
                "-m",
                message,
            ],
        )

        # 7. 再读取状态
        after = await self.inspector.inspect(worktree)

        if after.branch_name != workspace.branch_name:
            raise GitWorkspaceConflictError("Commit 后 Branch 异常")

        if after.head_sha == parent_sha:
            raise GitCommitError("Commit 后 HEAD 未发生变化")

        if not after.working_tree.is_clean:
            raise GitCommitError("Commit 后 Worktree " "仍存在未提交修改")

        return GitCommitResult(
            workspace_id=(workspace.workspace_id),
            task_id=(workspace.task_id),
            proposal_id=(workspace.proposal_id),
            branch_name=(workspace.branch_name),
            commit_sha=(after.head_sha),
            parent_sha=(parent_sha),
            message=message,
        )
