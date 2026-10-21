from __future__ import annotations

from pathlib import Path

from agent_core.git.cli_client import CliGitClient
from agent_core.git.client import GitClient


class GitDiffCollector:
    def __init__(self, *, git_client: GitClient | None = None) -> None:
        self.git_client = git_client or CliGitClient()

    async def changed_files(
        self,
        *,
        repository_root: str | Path,
        base_commit_sha: str,
        head_commit_sha: str,
    ) -> list[str]:
        if base_commit_sha == head_commit_sha:
            return []
        result = await self.git_client.run(
            cwd=Path(repository_root).resolve(strict=True),
            args=["diff", "--name-only", base_commit_sha, head_commit_sha],
        )
        return sorted(
            {
                line.strip().replace("\\", "/")
                for line in result.stdout.splitlines()
                if line.strip()
            }
        )
