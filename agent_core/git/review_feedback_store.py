import os
from pathlib import Path

from agent_core.git.review_feedback_model import (
    ReviewFeedbackArtifact,
)


class ReviewFeedbackStoreError(ValueError):
    pass


class ReviewFeedbackStore:

    def __init__(
        self,
        repository_root: str | Path,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise ReviewFeedbackStoreError("repository_root 不是目录")

        self.store_dir = self.root / ".fe-agent" / "runtime" / "review_feedback"

    def _validate_id(
        self,
        artifact_id: str,
    ) -> None:

        if len(artifact_id) != 64 or any(
            char not in ("0123456789abcdef") for char in artifact_id
        ):
            raise ReviewFeedbackStoreError("artifact_id 非法")

    def _path(
        self,
        artifact_id: str,
    ) -> Path:

        self._validate_id(artifact_id)

        return self.store_dir / f"{artifact_id}.json"

    def save(
        self,
        artifact: ReviewFeedbackArtifact,
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
    ) -> ReviewFeedbackArtifact:

        path = self._path(artifact_id)

        if not path.is_file():
            raise ReviewFeedbackStoreError("Review Feedback " "Artifact 不存在")

        try:

            content = path.read_text(encoding="utf-8")

            artifact = ReviewFeedbackArtifact.model_validate_json(content)

        except Exception as exc:

            raise ReviewFeedbackStoreError(
                "Review Feedback " "Artifact 文件无效"
            ) from exc

        if artifact.artifact_id != artifact_id:
            raise ReviewFeedbackStoreError("Review Feedback " "Artifact ID 不一致")

        return artifact
