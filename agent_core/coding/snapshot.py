import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

from agent_core.context.model import TaskContextBundle


class FileSnapshot(BaseModel):
    relative_path: str

    sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    size_bytes: int = Field(ge=0)


class TaskFileSnapshot(BaseModel):
    task_id: str

    files: list[FileSnapshot] = Field(default_factory=list)


class SnapshotError(ValueError):
    pass


class FileSnapshotBuilder:

    def __init__(
        self,
        repository_root: str | Path,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise SnapshotError("repository_root 不是目录")

    def build(
        self,
        bundle: TaskContextBundle,
    ) -> TaskFileSnapshot:

        snapshots: list[FileSnapshot] = []

        seen_paths: set[str] = set()

        for item in bundle.code_context:

            relative_path = item.relative_path

            if relative_path in seen_paths:
                continue

            seen_paths.add(relative_path)

            path = self._resolve_file(relative_path)

            data = path.read_bytes()

            snapshots.append(
                FileSnapshot(
                    relative_path=relative_path,
                    sha256=hashlib.sha256(data).hexdigest(),
                    size_bytes=len(data),
                )
            )

        snapshots.sort(key=lambda item: item.relative_path)

        return TaskFileSnapshot(
            task_id=(bundle.task.task_id),
            files=snapshots,
        )

    def _resolve_file(
        self,
        relative_path: str,
    ) -> Path:

        if not relative_path:
            raise SnapshotError("文件路径不能为空")

        path = (self.root / relative_path).resolve(strict=True)

        if not path.is_relative_to(self.root):
            raise SnapshotError("文件位于仓库范围之外: " f"{relative_path}")

        if not path.is_file():
            raise SnapshotError("目标不是普通文件: " f"{relative_path}")

        return path


class FileSnapshotValidator:

    def __init__(
        self,
        repository_root: str | Path,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

    def validate(
        self,
        snapshot: TaskFileSnapshot,
    ) -> None:

        for file_snapshot in snapshot.files:

            path = (self.root / file_snapshot.relative_path).resolve(strict=True)

            if not path.is_relative_to(self.root):
                raise SnapshotError(
                    "文件超出仓库范围: " f"{file_snapshot.relative_path}"
                )

            if not path.is_file():
                raise SnapshotError("快照文件不存在: " f"{file_snapshot.relative_path}")

            data = path.read_bytes()

            current_hash = hashlib.sha256(data).hexdigest()

            if current_hash != file_snapshot.sha256:
                raise SnapshotError(
                    "文件版本已经发生变化: " f"{file_snapshot.relative_path}"
                )
