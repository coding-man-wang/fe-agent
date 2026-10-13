import asyncio
import hashlib
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
from agent_core.git.review_feedback_model import (
    ReviewFeedbackItem,
)


class GitHubReviewFeedbackError(RuntimeError):
    pass


class GitHubReviewFeedbackApiClient(Protocol):

    async def get_review_comments(
        self,
        *,
        repository: str,
        pull_request_number: int,
    ) -> list[dict[str, Any]]: ...

    async def get_conversation_comments(
        self,
        *,
        repository: str,
        pull_request_number: int,
    ) -> list[dict[str, Any]]: ...


class DefaultGitHubReviewFeedbackApiClient:

    def __init__(
        self,
        *,
        token: str | None = None,
        api_base_url: str = ("https://api.github.com"),
    ) -> None:

        self.token = token or os.environ.get("GITHUB_TOKEN")

        if not self.token:
            raise GitHubReviewFeedbackError("缺少 GITHUB_TOKEN")

        self.api_base_url = api_base_url.rstrip("/")

    async def get_review_comments(
        self,
        *,
        repository: str,
        pull_request_number: int,
    ) -> list[dict[str, Any]]:

        url = (
            f"{self.api_base_url}"
            f"/repos/{repository}"
            f"/pulls/{pull_request_number}"
            f"/comments"
        )

        return await asyncio.to_thread(
            self._get_json_list,
            url,
        )

    async def get_conversation_comments(
        self,
        *,
        repository: str,
        pull_request_number: int,
    ) -> list[dict[str, Any]]:

        url = (
            f"{self.api_base_url}"
            f"/repos/{repository}"
            f"/issues/{pull_request_number}"
            f"/comments"
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

            raise GitHubReviewFeedbackError(
                "GitHub Review Feedback "
                "API 请求失败: "
                f"HTTP {exc.code} "
                f"{detail}"
            ) from exc

        try:

            result = json.loads(content.decode("utf-8"))

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:

            raise GitHubReviewFeedbackError(
                "GitHub Review Feedback " "API 返回无效 JSON"
            ) from exc

        if not isinstance(
            result,
            list,
        ):
            raise GitHubReviewFeedbackError(
                "GitHub Review Feedback " "API 返回结构非法"
            )

        return result


class GitHubReviewFeedbackProvider:

    def __init__(
        self,
        *,
        api_client: GitHubReviewFeedbackApiClient | None = None,
    ) -> None:

        self.api_client = api_client or DefaultGitHubReviewFeedbackApiClient()

    async def collect(
        self,
        *,
        pull_request: PullRequestResult,
    ) -> list[ReviewFeedbackItem]:

        if pull_request.provider != "github":
            raise GitHubReviewFeedbackError("Pull Request " "不是 GitHub Provider")

        inline_comments = await self.api_client.get_review_comments(
            repository=(pull_request.repository),
            pull_request_number=(pull_request.number),
        )

        conversation_comments = await self.api_client.get_conversation_comments(
            repository=(pull_request.repository),
            pull_request_number=(pull_request.number),
        )

        items: list[ReviewFeedbackItem] = []

        for raw in inline_comments:
            item = self._parse_inline_comment(raw)

            if item is not None:
                items.append(item)

        for raw in conversation_comments:
            item = self._parse_conversation_comment(raw)

            if item is not None:
                items.append(item)

        items.sort(key=lambda item: (item.external_id))

        return items

    def _parse_inline_comment(
        self,
        raw: dict[str, Any],
    ) -> ReviewFeedbackItem | None:

        if not isinstance(
            raw,
            dict,
        ):
            raise GitHubReviewFeedbackError("GitHub Review Comment " "结构非法")

        external_id = raw.get("id")

        user = raw.get("user")

        body = raw.get("body")

        path = raw.get("path")

        commit_sha = raw.get("commit_id")

        line = raw.get("line")

        url = raw.get("html_url")

        if not isinstance(
            external_id,
            int,
        ):
            raise GitHubReviewFeedbackError("GitHub Review Comment " "id 非法")

        reviewer = self._extract_reviewer(user)

        if reviewer is None:
            return None

        if (
            not isinstance(
                body,
                str,
            )
            or not body.strip()
        ):
            return None

        if (
            not isinstance(
                path,
                str,
            )
            or not path
        ):
            raise GitHubReviewFeedbackError("GitHub Review Comment " "path 非法")

        normalized_line = (
            line
            if isinstance(
                line,
                int,
            )
            else None
        )

        normalized_commit = (
            commit_sha
            if isinstance(
                commit_sha,
                str,
            )
            and commit_sha
            else None
        )

        external_id_text = str(external_id)

        return ReviewFeedbackItem(
            feedback_id=(
                self._calculate_feedback_id(
                    kind="inline_comment",
                    external_id=(external_id_text),
                )
            ),
            kind="inline_comment",
            reviewer=reviewer,
            body=body.strip(),
            relative_path=path,
            line=normalized_line,
            commit_sha=(normalized_commit),
            external_id=(external_id_text),
            url=self._optional_string(url),
        )

    def _parse_conversation_comment(
        self,
        raw: dict[str, Any],
    ) -> ReviewFeedbackItem | None:

        if not isinstance(
            raw,
            dict,
        ):
            raise GitHubReviewFeedbackError("GitHub Conversation Comment " "结构非法")

        external_id = raw.get("id")

        user = raw.get("user")

        body = raw.get("body")

        url = raw.get("html_url")

        if not isinstance(
            external_id,
            int,
        ):
            raise GitHubReviewFeedbackError("GitHub Conversation Comment " "id 非法")

        reviewer = self._extract_reviewer(user)

        if reviewer is None:
            return None

        if (
            not isinstance(
                body,
                str,
            )
            or not body.strip()
        ):
            return None

        external_id_text = str(external_id)

        return ReviewFeedbackItem(
            feedback_id=(
                self._calculate_feedback_id(
                    kind=("conversation_comment"),
                    external_id=(external_id_text),
                )
            ),
            kind=("conversation_comment"),
            reviewer=reviewer,
            body=body.strip(),
            relative_path=None,
            line=None,
            commit_sha=None,
            external_id=(external_id_text),
            url=self._optional_string(url),
        )

    def _extract_reviewer(
        self,
        user: Any,
    ) -> str | None:

        if not isinstance(
            user,
            dict,
        ):
            return None

        login = user.get("login")

        if (
            not isinstance(
                login,
                str,
            )
            or not login
        ):
            return None

        return login

    def _calculate_feedback_id(
        self,
        *,
        kind: str,
        external_id: str,
    ) -> str:

        raw = f"{kind}:{external_id}"

        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _optional_string(
        self,
        value: Any,
    ) -> str | None:

        if (
            isinstance(
                value,
                str,
            )
            and value
        ):
            return value

        return None
