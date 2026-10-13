from pathlib import Path

from agent_core.git.change_set_model import (
    GitChangeSet,
    GitFileChange,
)
from agent_core.git.client import (
    GitClient,
)
from agent_core.git.cli_client import (
    CliGitClient,
)
from agent_core.git.errors import (
    GitWorkspaceConflictError,
)
from agent_core.git.workspace_model import (
    GitWorkspaceRecord,
)


class GitChangeSetError(ValueError):
    pass


class GitChangeSetInspector:

    def __init__(
        self,
        *,
        git_client: GitClient | None = None,
    ) -> None:

        self.git_client = git_client or CliGitClient()

    async def inspect(
        self,
        *,
        workspace: GitWorkspaceRecord,
    ) -> GitChangeSet:

        if workspace.status != "ready":
            raise GitWorkspaceConflictError("Workspace 不是 ready 状态")

        root = Path(workspace.worktree_path).resolve()

        if not root.is_dir():
            raise GitWorkspaceConflictError("Worktree 不存在")

        result = await self.git_client.run(
            cwd=root,
            args=[
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
        )

        changes: list[GitFileChange] = []

        for line in result.stdout.splitlines():
            if not line:
                continue

            changes.append(self._parse_status_line(line))

        return GitChangeSet(files=changes)

    def _parse_status_line(
        self,
        line: str,
    ) -> GitFileChange:

        if len(line) < 4:
            raise GitChangeSetError("无法解析 Git status: " f"{line!r}")

        status = line[:2]

        path_text = line[3:]

        if status == "??":
            return GitFileChange(
                relative_path=path_text,
                kind="untracked",
                raw_status=status,
            )

        # porcelain v1:
        # R  old -> new
        # RM old -> new
        if "R" in status:
            return GitFileChange(
                relative_path=path_text,
                kind="renamed",
                raw_status=status,
            )

        if "D" in status:
            return GitFileChange(
                relative_path=path_text,
                kind="deleted",
                raw_status=status,
            )

        if "A" in status:
            return GitFileChange(
                relative_path=path_text,
                kind="added",
                raw_status=status,
            )

        if "M" in status:
            return GitFileChange(
                relative_path=path_text,
                kind="modified",
                raw_status=status,
            )

        return GitFileChange(
            relative_path=path_text,
            kind="other",
            raw_status=status,
        )
