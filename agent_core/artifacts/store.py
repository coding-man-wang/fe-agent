import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from agent_core.artifacts.model import ArtifactType, TaskArtifact
from collections.abc import Collection


class JsonTaskArtifactStore:
    """简单、可恢复、可审计的 Artifact Store；每个 Artifact 一个 JSON 文件。"""

    def __init__(self, *, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, artifact: TaskArtifact) -> Path:
        task_dir = self.root / artifact.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        path = (
            task_dir
            / f"{artifact.created_at.timestamp():.6f}-{artifact.artifact_id}.json"
        )
        self._atomic_write(path, artifact.model_dump_json(indent=2))
        return path

    def put_model(
        self,
        *,
        task_id: str,
        artifact_type: ArtifactType,
        model,
        depends_on_task_ids: list[str] | None = None,
    ) -> TaskArtifact:
        artifact = TaskArtifact(
            task_id=task_id,
            artifact_type=artifact_type,
            depends_on_task_ids=list(depends_on_task_ids or []),
            payload=model.model_dump(mode="json"),
        )
        self.put(artifact)
        return artifact

    def list(self, *, task_id: str | None = None) -> list[TaskArtifact]:
        if task_id is not None:
            task_dir = self.root / task_id
            if not task_dir.exists():
                return []
            files = list(task_dir.glob("*.json"))
        else:
            files = list(self.root.glob("*/*.json"))
        artifacts: list[TaskArtifact] = []
        for path in files:
            try:
                artifacts.append(
                    TaskArtifact.model_validate_json(path.read_text("utf-8"))
                )
            except (OSError, ValueError):
                continue
        artifacts.sort(key=lambda item: item.created_at)
        return artifacts

    def latest(
        self,
        *,
        task_id: str,
        artifact_type: ArtifactType,
    ) -> TaskArtifact | None:
        matches = [
            item
            for item in self.list(task_id=task_id)
            if item.artifact_type == artifact_type
        ]
        return matches[-1] if matches else None

    def seed_from(
        self,
        *,
        source: "JsonTaskArtifactStore",
        artifact_types: Collection[ArtifactType],
    ) -> int:
        """
        把另一个 Artifact Store 中指定类型的
        Artifact 复制到当前 Store。

        主要用于：

            baseline artifacts
                  ↓
            run-scoped artifacts

        注意：

        - target 必须为空；
        - source 和 target 不能是同一个目录；
        - Execution Artifact 不应默认跨 Run 复制。
        """

        if self.root == source.root:
            raise ValueError("不能从自身 seed Artifact Store")

        if self.list():
            raise ValueError("目标 Artifact Store 必须为空")

        allowed_types = set(artifact_types)

        copied = 0

        for artifact in source.list():
            if artifact.artifact_type not in allowed_types:
                continue

            # 保留原 artifact_id / created_at，
            # 因为这是已有历史证据的快照，
            # 不是重新生成 Artifact。
            self.put(artifact)
            copied += 1

        return copied

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
