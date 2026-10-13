import os
from pathlib import Path

from agent_core.git.follow_up_task_model import (
    FollowUpTaskArtifact,
)


class FollowUpTaskStoreError(ValueError):
    pass


class FollowUpTaskStore:

    def __init__(
        self,
        repository_root: str | Path,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise FollowUpTaskStoreError("repository_root 不是目录")

        self.store_dir = self.root / ".fe-agent" / "runtime" / "follow_up_tasks"

    def _validate_id(
        self,
        artifact_id: str,
    ) -> None:

        if len(artifact_id) != 64 or any(
            char not in "0123456789abcdef" for char in artifact_id
        ):
            raise FollowUpTaskStoreError("artifact_id 非法")

    def _path(
        self,
        artifact_id: str,
    ) -> Path:

        self._validate_id(artifact_id)

        return self.store_dir / f"{artifact_id}.json"

    def save(
        self,
        artifact: FollowUpTaskArtifact,
    ) -> None:

        path = self._path(artifact.artifact_id)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = path.with_suffix(".tmp")

        content = artifact.model_dump_json(indent=2)

        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(content)

            file.flush()

            os.fsync(file.fileno())

        os.replace(
            temp_path,
            path,
        )

    def load(
        self,
        artifact_id: str,
    ) -> FollowUpTaskArtifact:

        path = self._path(artifact_id)

        if not path.is_file():
            raise FollowUpTaskStoreError("Follow-up Task " "Artifact 不存在")

        try:

            content = path.read_text(encoding="utf-8")

            artifact = FollowUpTaskArtifact.model_validate_json(content)

        except Exception as exc:

            raise FollowUpTaskStoreError("Follow-up Task " "Artifact 文件无效") from exc

        if artifact.artifact_id != artifact_id:
            raise FollowUpTaskStoreError("Follow-up Task " "Artifact ID 不一致")

        return artifact
