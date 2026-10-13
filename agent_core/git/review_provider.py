from typing import Protocol

from agent_core.git.pull_request_model import (
    PullRequestResult,
)
from agent_core.git.review_model import (
    PullRequestReviewStatus,
)


class PullRequestReviewProvider(Protocol):

    async def get_status(
        self,
        *,
        pull_request: PullRequestResult,
    ) -> PullRequestReviewStatus: ...
