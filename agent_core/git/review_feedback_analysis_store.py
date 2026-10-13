import os
from pathlib import Path

from agent_core.git.review_feedback_analysis_model import (
    ReviewFeedbackAnalysisArtifact,
)


class ReviewFeedbackAnalysisStoreError(ValueError):
    pass


class ReviewFeedbackAnalysisStore:

    def __init__(
        self,
        repository_root: str | Path,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise (ReviewFeedbackAnalysisStoreError("repository_root 不是目录"))

        self.store_dir = (
            self.root / ".fe-agent" / "runtime" / "review_feedback_analysis"
        )

    def _validate_id(
        self,
        analysis_id: str,
    ) -> None:

        if len(analysis_id) != 64 or any(
            char not in "0123456789abcdef" for char in analysis_id
        ):
            raise (ReviewFeedbackAnalysisStoreError("analysis_id 非法"))

    def _path(
        self,
        analysis_id: str,
    ) -> Path:

        self._validate_id(analysis_id)

        return self.store_dir / f"{analysis_id}.json"

    def save(
        self,
        artifact: ReviewFeedbackAnalysisArtifact,
    ) -> None:

        path = self._path(artifact.analysis_id)

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
        analysis_id: str,
    ) -> ReviewFeedbackAnalysisArtifact:

        path = self._path(analysis_id)

        if not path.is_file():
            raise (
                ReviewFeedbackAnalysisStoreError("Review Feedback " "Analysis 不存在")
            )

        try:

            content = path.read_text(encoding="utf-8")

            artifact = ReviewFeedbackAnalysisArtifact.model_validate_json(content)

        except Exception as exc:

            raise (
                ReviewFeedbackAnalysisStoreError("Review Feedback " "Analysis 文件无效")
            ) from exc

        if artifact.analysis_id != analysis_id:
            raise (
                ReviewFeedbackAnalysisStoreError(
                    "Review Feedback " "Analysis ID 不一致"
                )
            )

        return artifact
