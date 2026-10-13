import hashlib
import json
from pathlib import Path

from agent_core.git.analysis_snapshot_model import (
    AnalysisSnapshotRecord,
)
from agent_core.git.analysis_snapshot_store import (
    AnalysisSnapshotStore,
)
from agent_core.git.cli_client import (
    CliGitClient,
)
from agent_core.git.client import (
    GitClient,
)
from agent_core.git.repository import (
    GitRepositoryInspector,
)


class AnalysisSnapshotError(ValueError):
    pass


class AnalysisSnapshotManager:

    def __init__(
        self,
        *,
        repository_root: str | Path,
        snapshot_root: str | Path,
        git_client: GitClient | None = None,
        inspector: GitRepositoryInspector | None = None,
        store: AnalysisSnapshotStore | None = None,
    ) -> None:

        self.repository_root = Path(repository_root).resolve(strict=True)

        self.snapshot_root = Path(snapshot_root).resolve()

        self.git_client = git_client or CliGitClient()

        self.inspector = inspector or GitRepositoryInspector(git_client=self.git_client)

        self.store = store or AnalysisSnapshotStore(self.repository_root)

        self._validate_snapshot_root()

    async def create(
        self,
        *,
        task_id: str,
        base_commit_sha: str,
    ) -> AnalysisSnapshotRecord:

        self._validate_sha(base_commit_sha)

        await self._validate_commit_exists(base_commit_sha)

        snapshot_id = self.calculate_snapshot_id(
            task_id=task_id,
            base_commit_sha=(base_commit_sha),
        )

        if self.store.exists(snapshot_id):
            return await self._reuse(snapshot_id)

        snapshot_path = self.snapshot_root / snapshot_id

        if snapshot_path.exists():
            raise AnalysisSnapshotError(
                "Snapshot 路径已经存在，" "但没有对应持久化记录"
            )

        record = AnalysisSnapshotRecord(
            snapshot_id=snapshot_id,
            task_id=task_id,
            repository_root=str(self.repository_root),
            base_commit_sha=(base_commit_sha),
            snapshot_path=str(snapshot_path),
            status="creating",
            error=None,
        )

        self.store.save(record)

        try:

            self.snapshot_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            await self.git_client.run(
                cwd=self.repository_root,
                args=[
                    "worktree",
                    "add",
                    "--detach",
                    str(snapshot_path),
                    base_commit_sha,
                ],
            )

            await self._validate_snapshot(record)

            ready = record.model_copy(
                update={
                    "status": "ready",
                    "error": None,
                }
            )

            self.store.save(ready)

            return ready

        except Exception as exc:

            failed = record.model_copy(
                update={
                    "status": "failed",
                    "error": (f"{type(exc).__name__}: " f"{exc}"),
                }
            )

            self.store.save(failed)

            raise

    async def _reuse(
        self,
        snapshot_id: str,
    ) -> AnalysisSnapshotRecord:

        record = self.store.load(snapshot_id)

        if record.status == "ready":

            await self._validate_snapshot(record)

            return record

        if record.status == "creating":

            path = Path(record.snapshot_path)

            if path.exists():

                await self._validate_snapshot(record)

                ready = record.model_copy(
                    update={
                        "status": "ready",
                        "error": None,
                    }
                )

                self.store.save(ready)

                return ready

            raise AnalysisSnapshotError("Snapshot 处于 creating，" "但 Worktree 不存在")

        if record.status == "failed":

            raise AnalysisSnapshotError("Snapshot 创建曾失败，" "禁止自动复用")

        if record.status == "removed":

            raise AnalysisSnapshotError("Snapshot 已移除")

        raise AnalysisSnapshotError("未知 Snapshot 状态")

    async def _validate_snapshot(
        self,
        record: AnalysisSnapshotRecord,
    ) -> None:

        path = Path(record.snapshot_path)

        if not path.is_dir():

            raise AnalysisSnapshotError("Snapshot Worktree 不存在")

        state = await self.inspector.inspect(path)

        actual_root = Path(state.repository_root).resolve()

        if actual_root != path.resolve():

            raise AnalysisSnapshotError("Snapshot Repository Root " "不一致")

        if not state.is_detached_head:

            raise AnalysisSnapshotError("Analysis Snapshot " "必须使用 Detached HEAD")

        if state.head_sha != record.base_commit_sha:

            raise AnalysisSnapshotError("Snapshot HEAD " "与 Base Commit 不一致")

        if state.working_tree.tracked_changes:

            raise AnalysisSnapshotError("Snapshot 存在已跟踪文件修改")

        if state.working_tree.untracked_files:

            raise AnalysisSnapshotError("Snapshot 存在未跟踪文件")

    async def _validate_commit_exists(
        self,
        commit_sha: str,
    ) -> None:

        result = await self.git_client.run(
            cwd=self.repository_root,
            args=[
                "cat-file",
                "-e",
                f"{commit_sha}^{{commit}}",
            ],
            allow_failure=True,
        )

        if result.exit_code != 0:

            raise AnalysisSnapshotError("Base Commit 不存在: " f"{commit_sha}")

    def _validate_snapshot_root(
        self,
    ) -> None:

        if (
            self.snapshot_root == self.repository_root
            or self.snapshot_root.is_relative_to(self.repository_root)
        ):
            raise ValueError("Snapshot Root " "不能位于 Repository Root 内")

    @staticmethod
    def calculate_snapshot_id(
        *,
        task_id: str,
        base_commit_sha: str,
    ) -> str:

        payload = json.dumps(
            {
                "task_id": task_id,
                "base_commit_sha": base_commit_sha,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_sha(
        value: str,
    ) -> None:

        if len(value) != 40:

            raise ValueError("非法 Base Commit SHA")

        if any(char not in "0123456789abcdef" for char in value):

            raise ValueError("非法 Base Commit SHA")
