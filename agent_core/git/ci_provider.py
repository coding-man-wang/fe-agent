from typing import Protocol

from agent_core.git.ci_model import (
    PullRequestCIStatus,
)
from agent_core.git.pull_request_model import (
    PullRequestResult,
)


class PullRequestCIProvider(Protocol):

    async def get_status(
        self,
        *,
        pull_request: PullRequestResult,
    ) -> PullRequestCIStatus: ...
