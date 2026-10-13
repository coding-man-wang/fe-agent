import asyncio
import json
import os
from typing import (
    Any,
    Protocol,
)
from urllib.error import (
    HTTPError,
)
from urllib.request import (
    Request,
    urlopen,
)

from agent_core.git.pull_request_model import (
    PullRequestResult,
)
from agent_core.git.review_model import (
    PullRequestReviewEntry,
    PullRequestReviewStatus,
)


class GitHubReviewError(RuntimeError):
    pass


class GitHubReviewApiClient(Protocol):

    async def get_reviews(
        self,
        *,
        repository: str,
        pull_request_number: int,
    ) -> list[dict[str, Any]]: ...


class DefaultGitHubReviewApiClient:

    def __init__(
        self,
        *,
        token: str | None = None,
        api_base_url: str = ("https://api.github.com"),
    ) -> None:

        self.token = token or os.environ.get("GITHUB_TOKEN")

        if not self.token:
            raise GitHubReviewError("缺少 GITHUB_TOKEN")

        self.api_base_url = api_base_url.rstrip("/")

    async def get_reviews(
        self,
        *,
        repository: str,
        pull_request_number: int,
    ) -> list[dict[str, Any]]:

        url = (
            f"{self.api_base_url}"
            f"/repos/{repository}"
            f"/pulls/{pull_request_number}"
            f"/reviews"
        )

        return await asyncio.to_thread(
            self._get_json_list,
            url,
        )

    def _get_json_list(
        self,
        url: str,
    ) -> list[dict[str, Any]]:

        request = Request(
            url=url,
            method="GET",
            headers={
                "Accept": ("application/vnd.github+json"),
                "Authorization": (f"Bearer {self.token}"),
                "X-GitHub-Api-Version": ("2022-11-28"),
            },
        )

        try:

            with urlopen(
                request,
                timeout=30,
            ) as response:

                content = response.read()

        except HTTPError as exc:

            try:
                detail = exc.read().decode(
                    "utf-8",
                    errors="replace",
                )

            except Exception:
                detail = ""

            raise GitHubReviewError(
                "GitHub Review API 请求失败: " f"HTTP {exc.code} " f"{detail}"
            ) from exc

        try:

            result = json.loads(content.decode("utf-8"))

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:

            raise GitHubReviewError("GitHub Review API " "返回无效 JSON") from exc

        if not isinstance(
            result,
            list,
        ):
            raise GitHubReviewError("GitHub Review API " "返回结构非法")

        return result


class GitHubReviewProvider:

    def __init__(
        self,
        *,
        api_client: GitHubReviewApiClient | None = None,
    ) -> None:

        self.api_client = api_client or DefaultGitHubReviewApiClient()

    async def get_status(
        self,
        *,
        pull_request: PullRequestResult,
    ) -> PullRequestReviewStatus:

        if pull_request.provider != "github":
            raise GitHubReviewError("Pull Request " "不是 GitHub Provider")

        raw_reviews = await self.api_client.get_reviews(
            repository=(pull_request.repository),
            pull_request_number=(pull_request.number),
        )

        latest_by_reviewer: dict[
            str,
            dict[str, Any],
        ] = {}

        for item in raw_reviews:

            if not isinstance(
                item,
                dict,
            ):
                raise GitHubReviewError("GitHub Review 结构非法")

            user = item.get("user")

            if not isinstance(
                user,
                dict,
            ):
                continue

            login = user.get("login")

            if (
                not isinstance(
                    login,
                    str,
                )
                or not login
            ):
                continue

            state = item.get("state")

            if not isinstance(
                state,
                str,
            ):
                continue

            normalized_state = state.upper()

            # COMMENTED 不改变审批结论
            if normalized_state == "COMMENTED":
                continue

            # PENDING 也不作为最终已提交 Review
            if normalized_state == "PENDING":
                continue

            latest_by_reviewer[login] = item

        reviews: list[PullRequestReviewEntry] = []

        approvals = 0
        changes_requested = 0

        for reviewer, item in latest_by_reviewer.items():

            raw_state = str(item.get("state")).upper()

            if raw_state == "APPROVED":
                state = "approved"
                approvals += 1

            elif raw_state == "CHANGES_REQUESTED":
                state = "changes_requested"
                changes_requested += 1

            elif raw_state == "DISMISSED":
                # dismissed 表示之前的 Review
                # 已失效，不再阻塞。
                continue

            else:
                continue

            reviews.append(
                PullRequestReviewEntry(
                    reviewer=reviewer,
                    state=state,
                    raw_state=raw_state,
                )
            )

        reviews.sort(key=lambda item: (item.reviewer))

        if changes_requested > 0:
            overall = "changes_requested"

        elif approvals > 0:
            overall = "approved"

        else:
            overall = "pending"

        return PullRequestReviewStatus(
            provider="github",
            repository=(pull_request.repository),
            pull_request_number=(pull_request.number),
            commit_sha=(pull_request.commit_sha),
            status=overall,
            approvals=approvals,
            changes_requested=(changes_requested),
            reviews=reviews,
        )
