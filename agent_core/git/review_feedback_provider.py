from typing import Protocol

from agent_core.git.pull_request_model import (
    PullRequestResult,
)
from agent_core.git.review_feedback_model import (
    ReviewFeedbackItem,
)


class ReviewFeedbackProvider(Protocol):

    async def collect(
        self,
        *,
        pull_request: PullRequestResult,
    ) -> list[ReviewFeedbackItem]: ...
