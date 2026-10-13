import hashlib
import json

from agent_core.git.pull_request_model import (
    PullRequestResult,
)
from agent_core.git.review_feedback_model import (
    ReviewFeedbackArtifact,
    ReviewFeedbackItem,
)


class ReviewFeedbackError(ValueError):
    pass


class ReviewFeedbackService:

    def build_artifact(
        self,
        *,
        pull_request: PullRequestResult,
        items: list[ReviewFeedbackItem],
    ) -> ReviewFeedbackArtifact:

        if pull_request.number < 1:
            raise ReviewFeedbackError("Pull Request number 非法")

        deduplicated = self._deduplicate(items)

        artifact_id = self._calculate_artifact_id(
            pull_request=(pull_request),
            items=deduplicated,
        )

        return ReviewFeedbackArtifact(
            artifact_id=artifact_id,
            pull_request_provider=(pull_request.provider),
            repository=(pull_request.repository),
            pull_request_number=(pull_request.number),
            task_id=(pull_request.task_id),
            proposal_id=(pull_request.proposal_id),
            commit_sha=(pull_request.commit_sha),
            items=deduplicated,
        )

    def _deduplicate(
        self,
        items: list[ReviewFeedbackItem],
    ) -> list[ReviewFeedbackItem]:

        result: dict[
            str,
            ReviewFeedbackItem,
        ] = {}

        for item in items:

            existing = result.get(item.feedback_id)

            if existing is not None and existing != item:
                raise ReviewFeedbackError("相同 feedback_id " "对应不同内容")

            result[item.feedback_id] = item

        return sorted(
            result.values(),
            key=lambda item: (item.feedback_id),
        )

    def _calculate_artifact_id(
        self,
        *,
        pull_request: PullRequestResult,
        items: list[ReviewFeedbackItem],
    ) -> str:

        payload = {
            "provider": (pull_request.provider),
            "repository": (pull_request.repository),
            "pull_request_number": (pull_request.number),
            "task_id": (pull_request.task_id),
            "proposal_id": (pull_request.proposal_id),
            "commit_sha": (pull_request.commit_sha),
            "feedback_ids": [item.feedback_id for item in items],
        }

        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()
