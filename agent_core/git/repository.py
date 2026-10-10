from pathlib import Path

from agent_core.git.cli_client import (
    CliGitClient,
)
from agent_core.git.client import (
    GitClient,
)
from agent_core.git.errors import (
    GitCommandError,
    NotGitRepositoryError,
)
from agent_core.git.model import (
    GitRepositoryState,
    GitWorkingTreeStatus,
)


class GitRepositoryInspector:
    """
    读取 Git Repository 当前状态。

    只读组件，不产生 Git Repository 副作用。
    """

    def __init__(
            self,
            *,
            git_client: GitClient | None = None,
    ) -> None:
        self.git_client = git_client or CliGitClient()

    async def inspect(
            self,
            repository_path: str | Path,
    ) -> GitRepositoryState:

        requested_path = Path(repository_path).resolve()

        if not requested_path.exists():
            raise NotGitRepositoryError("Repository path 不存在: " f"{requested_path}")

        if not requested_path.is_dir():
            raise NotGitRepositoryError(
                "Repository path 不是目录: " f"{requested_path}"
            )

        repository_root = await self._resolve_root(requested_path)

        head_sha = await self._read_head(repository_root)

        branch_name = await self._read_branch(repository_root)

        working_tree = await self._read_status(repository_root)

        return GitRepositoryState(
            repository_root=str(repository_root),
            head_sha=head_sha,
            branch_name=branch_name,
            is_detached_head=(branch_name is None),
            working_tree=working_tree,
        )

    async def _resolve_root(
            self,
            cwd: Path,
    ) -> Path:

        result = await self.git_client.run(
            cwd=cwd,
            args=[
                "rev-parse",
                "--show-toplevel",
            ],
            allow_failure=True,
        )

        if result.exit_code != 0:
            raise NotGitRepositoryError("目标路径不是有效 Git Repository: " f"{cwd}")

        root_text = result.stdout.strip()

        if not root_text:
            raise NotGitRepositoryError("Git 未返回 Repository Root")

        root = Path(root_text).resolve()

        if not root.is_dir():
            raise NotGitRepositoryError(
                "Git 返回的 Repository Root " "不存在: " f"{root}"
            )

        return root

    async def _read_head(
            self,
            root: Path,
    ) -> str:

        result = await self.git_client.run(
            cwd=root,
            args=[
                "rev-parse",
                "HEAD",
            ],
        )

        sha = result.stdout.strip()

        if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
            raise GitCommandError(
                args=[
                    "rev-parse",
                    "HEAD",
                ],
                exit_code=result.exit_code,
                stderr=("Git 返回了非法 HEAD SHA: " f"{sha}"),
            )

        return sha

    async def _read_branch(
            self,
            root: Path,
    ) -> str | None:

        result = await self.git_client.run(
            cwd=root,
            args=[
                "symbolic-ref",
                "--quiet",
                "--short",
                "HEAD",
            ],
            allow_failure=True,
        )

        # symbolic-ref 在 Detached HEAD
        # 通常返回 exit code 1。
        if result.exit_code == 1:
            return None

        if result.exit_code != 0:
            raise GitCommandError(
                args=[
                    "symbolic-ref",
                    "--quiet",
                    "--short",
                    "HEAD",
                ],
                exit_code=result.exit_code,
                stderr=result.stderr,
            )

        branch = result.stdout.strip()

        return branch or None

    async def _read_status(
            self,
            root: Path,
    ) -> GitWorkingTreeStatus:

        result = await self.git_client.run(
            cwd=root,
            args=[
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
        )

        tracked_changes: list[str] = []
        untracked_files: list[str] = []

        for raw_line in result.stdout.splitlines():
            if not raw_line:
                continue

            if raw_line.startswith("?? "):
                untracked_files.append(raw_line[3:])

                continue

            tracked_changes.append(raw_line)

        return GitWorkingTreeStatus(
            tracked_changes=(tracked_changes),
            untracked_files=(untracked_files),
        )
