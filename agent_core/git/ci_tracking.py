import asyncio

from agent_core.git.ci_model import (
    PullRequestCIStatus,
)
from agent_core.git.ci_provider import (
    PullRequestCIProvider,
)
from agent_core.git.ci_tracking_model import (
    CITrackingResult,
)
from agent_core.git.pull_request_model import (
    PullRequestResult,
)


class CITrackingError(RuntimeError):
    pass


class CITrackingWorkflow:

    def __init__(
        self,
        *,
        provider: PullRequestCIProvider,
        max_attempts: int = 10,
        poll_interval_seconds: float = 5.0,
    ) -> None:

        if max_attempts < 1:
            raise ValueError("max_attempts 必须 >= 1")

        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds 不能小于 0")

        self.provider = provider
        self.max_attempts = max_attempts
        self.poll_interval_seconds = poll_interval_seconds

    async def track(
        self,
        *,
        pull_request: PullRequestResult,
    ) -> CITrackingResult:

        final_status: PullRequestCIStatus | None = None

        for attempt in range(
            1,
            self.max_attempts + 1,
        ):

            status = await self.provider.get_status(pull_request=pull_request)

            final_status = status

            if status.status == "succeeded":
                return CITrackingResult(
                    outcome="succeeded",
                    attempts=attempt,
                    final_status=status,
                )

            if status.status == "failed":
                return CITrackingResult(
                    outcome="failed",
                    attempts=attempt,
                    final_status=status,
                )

            if status.status == "no_checks":
                return CITrackingResult(
                    outcome="no_checks",
                    attempts=attempt,
                    final_status=status,
                )

            if attempt < self.max_attempts and self.poll_interval_seconds > 0:
                await asyncio.sleep(self.poll_interval_seconds)

        if final_status is None:
            raise CITrackingError("CI Tracking 未获得任何状态")

        return CITrackingResult(
            outcome="timeout",
            attempts=self.max_attempts,
            final_status=final_status,
        )
