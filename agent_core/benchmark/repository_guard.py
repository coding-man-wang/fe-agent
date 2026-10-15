from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
)

from agent_core.benchmark.case_model import (
    BenchmarkCase,
)
from agent_core.git.cli_client import (
    CliGitClient,
)
from agent_core.git.client import (
    GitClient,
)


class BenchmarkRepositoryError(ValueError):
    pass


class BenchmarkRepositoryState(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
    )

    repository_root: str

    expected_commit_sha: str

    actual_commit_sha: str

    clean: bool


class BenchmarkRepositoryGuard:

    def __init__(
        self,
        *,
        repository_root: str | Path,
        git_client: GitClient | None = None,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise BenchmarkRepositoryError("repository_root 不是目录")

        self.git_client = git_client or CliGitClient()

    async def inspect(
        self,
        *,
        case: BenchmarkCase,
    ) -> BenchmarkRepositoryState:

        actual_head = await self._read_head()

        clean = await self._is_clean()

        return BenchmarkRepositoryState(
            repository_root=str(self.root),
            expected_commit_sha=(case.base_commit_sha),
            actual_commit_sha=(actual_head),
            clean=clean,
        )

    async def require_ready(
        self,
        *,
        case: BenchmarkCase,
    ) -> BenchmarkRepositoryState:

        state = await self.inspect(case=case)

        if state.actual_commit_sha != state.expected_commit_sha:
            raise BenchmarkRepositoryError(
                "Benchmark Repository HEAD " "与 Case Base Commit 不一致"
            )

        if not state.clean:
            raise BenchmarkRepositoryError("Benchmark Repository " "存在未提交修改")

        return state

    async def _read_head(
        self,
    ) -> str:

        result = await self.git_client.run(
            cwd=self.root,
            args=[
                "rev-parse",
                "HEAD",
            ],
        )

        sha = result.stdout.strip()

        if len(sha) != 40:
            raise BenchmarkRepositoryError("无法读取合法 Git HEAD")

        if any(char not in "0123456789abcdef" for char in sha):
            raise BenchmarkRepositoryError("Git HEAD 格式非法")

        return sha

    async def _is_clean(
        self,
    ) -> bool:

        result = await self.git_client.run(
            cwd=self.root,
            args=[
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
        )

        return result.stdout.strip() == ""
