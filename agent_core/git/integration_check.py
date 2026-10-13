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
from agent_core.git.integration_model import (
    GitIntegrationCheckResult,
)
from agent_core.git.workspace_model import (
    GitWorkspaceRecord,
)


class GitIntegrationCheckError(RuntimeError):
    pass


class GitIntegrationCheckService:

    def __init__(
        self,
        *,
        git_client: GitClient | None = None,
    ) -> None:

        self.git_client = git_client or CliGitClient()

    async def check(
        self,
        *,
        workspace: GitWorkspaceRecord,
        commit: GitCommitResult,
        target_branch: str,
    ) -> GitIntegrationCheckResult:

        root = Path(workspace.repository_root).resolve()

        if not root.is_dir():
            raise GitWorkspaceConflictError("Source Repository 不存在")

        if workspace.status != "ready":
            raise GitWorkspaceConflictError("Workspace 不是 ready 状态")

        if commit.workspace_id != workspace.workspace_id:
            raise GitIntegrationCheckError("Commit 不属于当前 Workspace")

        if commit.task_id != workspace.task_id:
            raise GitIntegrationCheckError("Commit Task ID 不一致")

        if commit.proposal_id != workspace.proposal_id:
            raise GitIntegrationCheckError("Commit Proposal ID 不一致")

        target_branch = target_branch.strip()

        if not target_branch:
            raise GitIntegrationCheckError("target_branch 不能为空")

        # --------------------------
        # 1. 解析目标分支最新 HEAD
        # --------------------------

        target_result = await self.git_client.run(
            cwd=root,
            args=[
                "rev-parse",
                target_branch,
            ],
        )

        target_head_sha = target_result.stdout.strip()

        if len(target_head_sha) != 40:
            raise GitIntegrationCheckError("目标分支 HEAD 非法")

        base_commit_sha = workspace.base_commit_sha

        agent_commit_sha = commit.commit_sha

        # --------------------------
        # 2. 目标分支没前进
        # --------------------------

        if target_head_sha == base_commit_sha:
            return GitIntegrationCheckResult(
                target_branch=target_branch,
                base_commit_sha=(base_commit_sha),
                target_head_sha=(target_head_sha),
                agent_commit_sha=(agent_commit_sha),
                status="up_to_date",
                target_advanced=False,
                conflicting_files=[],
            )

        # --------------------------
        # 3. 目标分支前进了
        #
        # 用 merge-tree 检查是否存在冲突。
        # 不真正修改任何分支。
        # --------------------------

        result = await self.git_client.run(
            cwd=root,
            args=[
                "merge-tree",
                "--write-tree",
                target_head_sha,
                agent_commit_sha,
            ],
            allow_failure=True,
        )

        if result.exit_code == 0:
            return GitIntegrationCheckResult(
                target_branch=target_branch,
                base_commit_sha=(base_commit_sha),
                target_head_sha=(target_head_sha),
                agent_commit_sha=(agent_commit_sha),
                status="clean",
                target_advanced=True,
                conflicting_files=[],
            )

        conflicting_files = self._extract_conflicting_files(
            result.stdout + "\n" + result.stderr
        )

        return GitIntegrationCheckResult(
            target_branch=target_branch,
            base_commit_sha=(base_commit_sha),
            target_head_sha=(target_head_sha),
            agent_commit_sha=(agent_commit_sha),
            status="conflict",
            target_advanced=True,
            conflicting_files=(conflicting_files),
        )

    def _extract_conflicting_files(
        self,
        output: str,
    ) -> list[str]:

        files: set[str] = set()

        for line in output.splitlines():
            line = line.strip()

            if not line:
                continue

            if "CONFLICT" not in line:
                continue

            if " in " not in line:
                continue

            path = line.rsplit(
                " in ",
                1,
            )[-1].strip()

            if path:
                files.add(path)

        return sorted(files)
