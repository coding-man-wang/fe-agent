from typing import Protocol

from agent_core.git.pull_request_model import (
    PullRequestRequest,
    PullRequestResult,
)


class PullRequestProvider(Protocol):

    async def create(
        self,
        *,
        request: PullRequestRequest,
    ) -> PullRequestResult: ...
