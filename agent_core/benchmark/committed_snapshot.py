from __future__ import annotations

import re
from pathlib import Path

from agent_core.git.cli_client import CliGitClient
from agent_core.git.client import GitClient


class CommittedSnapshotError(RuntimeError):
    pass


class CommittedSnapshotWorktreeManager:
    """Materialize an immutable detached worktree for analysis at an exact commit.

    v1.1 must never build Context/Snapshot/PatchPlan from the mutable benchmark
    repository working tree and then apply that proposal to a clean Git worktree.
    Both sides must be derived from the same commit identity.
    """

    def __init__(
        self,
        *,
        repository_root: str | Path,
        snapshot_root: str | Path,
        git_client: GitClient | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve(strict=True)
        self.snapshot_root = Path(snapshot_root).resolve()
        self.git_client = git_client or CliGitClient()

    async def create(
        self,
        *,
        task_id: str,
        commit_sha: str,
    ) -> Path:
        self._validate_commit_sha(commit_sha)
        task_slug = re.sub(
            r"[^a-z0-9._-]+",
            "-",
            task_id.strip().lower(),
        ).strip("-._") or "task"

        target = self.snapshot_root / f"{task_slug}-{commit_sha[:12]}"
        if target.exists():
            raise CommittedSnapshotError(
                f"Committed snapshot 路径已存在: {target}"
            )

        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        await self.git_client.run(
            cwd=self.repository_root,
            args=[
                "worktree",
                "add",
                "--detach",
                str(target),
                commit_sha,
            ],
        )

        try:
            head = await self.git_client.run(
                cwd=target,
                args=["rev-parse", "HEAD"],
            )
            actual = head.stdout.strip().lower()
            if actual != commit_sha:
                raise CommittedSnapshotError(
                    "Committed snapshot HEAD 与请求 commit 不一致: "
                    f"expected={commit_sha}, actual={actual}"
                )
            return target.resolve(strict=True)
        except Exception:
            await self.remove(target, allow_missing=True)
            raise

    async def remove(
        self,
        path: str | Path,
        *,
        allow_missing: bool = False,
    ) -> None:
        target = Path(path).resolve()
        if not target.exists() and allow_missing:
            return

        result = await self.git_client.run(
            cwd=self.repository_root,
            args=["worktree", "remove", "--force", str(target)],
            allow_failure=True,
        )
        if result.exit_code != 0 and not allow_missing:
            raise CommittedSnapshotError(
                "无法移除 committed snapshot: " + result.stderr.strip()
            )

    @staticmethod
    def _validate_commit_sha(value: str) -> None:
        if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("commit_sha 必须是 40 位小写十六进制 SHA")
