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

from agent_core.git.ci_model import (
    CICheckResult,
    PullRequestCIStatus,
)
from agent_core.git.pull_request_model import (
    PullRequestResult,
)


class GitHubCIError(RuntimeError):
    pass


class GitHubCIApiClient(Protocol):

    async def get_combined_status(
        self,
        *,
        repository: str,
        commit_sha: str,
    ) -> dict[str, Any]: ...

    async def get_check_runs(
        self,
        *,
        repository: str,
        commit_sha: str,
    ) -> dict[str, Any]: ...


class DefaultGitHubCIApiClient:

    def __init__(
        self,
        *,
        token: str | None = None,
        api_base_url: str = ("https://api.github.com"),
    ) -> None:

        self.token = token or os.environ.get("GITHUB_TOKEN")

        if not self.token:
            raise GitHubCIError("缺少 GITHUB_TOKEN")

        self.api_base_url = api_base_url.rstrip("/")

    async def get_combined_status(
        self,
        *,
        repository: str,
        commit_sha: str,
    ) -> dict[str, Any]:

        url = (
            f"{self.api_base_url}"
            f"/repos/{repository}"
            f"/commits/{commit_sha}"
            f"/status"
        )

        return await asyncio.to_thread(
            self._get_json,
            url,
        )

    async def get_check_runs(
        self,
        *,
        repository: str,
        commit_sha: str,
    ) -> dict[str, Any]:

        url = (
            f"{self.api_base_url}"
            f"/repos/{repository}"
            f"/commits/{commit_sha}"
            f"/check-runs"
        )

        return await asyncio.to_thread(
            self._get_json,
            url,
        )

    def _get_json(
        self,
        url: str,
    ) -> dict[str, Any]:

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

            raise GitHubCIError(
                "GitHub CI API 请求失败: " f"HTTP {exc.code} " f"{detail}"
            ) from exc

        try:

            result = json.loads(content.decode("utf-8"))

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:

            raise GitHubCIError("GitHub CI API " "返回无效 JSON") from exc

        if not isinstance(
            result,
            dict,
        ):
            raise GitHubCIError("GitHub CI API " "返回结构非法")

        return result


class GitHubCIProvider:

    FAILURE_CONCLUSIONS = {
        "failure",
        "cancelled",
        "timed_out",
        "action_required",
        "startup_failure",
        "stale",
    }

    SUCCESS_CONCLUSIONS = {
        "success",
        "neutral",
        "skipped",
    }

    def __init__(
        self,
        *,
        api_client: GitHubCIApiClient | None = None,
    ) -> None:

        self.api_client = api_client or DefaultGitHubCIApiClient()

    async def get_status(
        self,
        *,
        pull_request: PullRequestResult,
    ) -> PullRequestCIStatus:

        if pull_request.provider != "github":
            raise GitHubCIError("Pull Request " "不是 GitHub Provider")

        combined = await self.api_client.get_combined_status(
            repository=(pull_request.repository),
            commit_sha=(pull_request.commit_sha),
        )

        check_runs = await self.api_client.get_check_runs(
            repository=(pull_request.repository),
            commit_sha=(pull_request.commit_sha),
        )

        checks: list[CICheckResult] = []

        checks.extend(self._parse_commit_statuses(combined))

        checks.extend(self._parse_check_runs(check_runs))

        overall = self._calculate_overall_status(checks)

        return PullRequestCIStatus(
            provider="github",
            repository=(pull_request.repository),
            pull_request_number=(pull_request.number),
            commit_sha=(pull_request.commit_sha),
            status=overall,
            checks=checks,
        )

    def _parse_commit_statuses(
        self,
        payload: dict[str, Any],
    ) -> list[CICheckResult]:

        raw_statuses = payload.get(
            "statuses",
            [],
        )

        if not isinstance(
            raw_statuses,
            list,
        ):
            raise GitHubCIError("GitHub combined status " "statuses 非法")

        results: list[CICheckResult] = []

        for item in raw_statuses:

            if not isinstance(
                item,
                dict,
            ):
                raise GitHubCIError("GitHub commit status " "结构非法")

            context = item.get("context")

            state = item.get("state")

            if (
                not isinstance(
                    context,
                    str,
                )
                or not context
            ):
                raise GitHubCIError("GitHub commit status " "context 非法")

            if not isinstance(
                state,
                str,
            ):
                raise GitHubCIError("GitHub commit status " "state 非法")

            results.append(
                CICheckResult(
                    source=("commit_status"),
                    name=context,
                    state=(self._normalize_commit_state(state)),
                    raw_status=state,
                    raw_conclusion=None,
                    url=self._optional_string(item.get("target_url")),
                )
            )

        return results

    def _parse_check_runs(
        self,
        payload: dict[str, Any],
    ) -> list[CICheckResult]:

        raw_runs = payload.get(
            "check_runs",
            [],
        )

        if not isinstance(
            raw_runs,
            list,
        ):
            raise GitHubCIError("GitHub check_runs 非法")

        results: list[CICheckResult] = []

        for item in raw_runs:

            if not isinstance(
                item,
                dict,
            ):
                raise GitHubCIError("GitHub check run " "结构非法")

            name = item.get("name")

            status = item.get("status")

            conclusion = item.get("conclusion")

            if (
                not isinstance(
                    name,
                    str,
                )
                or not name
            ):
                raise GitHubCIError("GitHub check run " "name 非法")

            if not isinstance(
                status,
                str,
            ):
                raise GitHubCIError("GitHub check run " "status 非法")

            normalized_state = self._normalize_check_run(
                status=status,
                conclusion=conclusion,
            )

            results.append(
                CICheckResult(
                    source="check_run",
                    name=name,
                    state=normalized_state,
                    raw_status=status,
                    raw_conclusion=(
                        conclusion
                        if isinstance(
                            conclusion,
                            str,
                        )
                        else None
                    ),
                    url=self._optional_string(
                        item.get("details_url") or item.get("html_url")
                    ),
                )
            )

        return results

    def _normalize_commit_state(
        self,
        state: str,
    ) -> str:

        normalized = state.lower()

        if normalized == "success":
            return "succeeded"

        if normalized in {
            "failure",
            "error",
        }:
            return "failed"

        return "pending"

    def _normalize_check_run(
        self,
        *,
        status: str,
        conclusion: Any,
    ) -> str:

        normalized_status = status.lower()

        if normalized_status != "completed":
            return "pending"

        if not isinstance(
            conclusion,
            str,
        ):
            return "pending"

        normalized_conclusion = conclusion.lower()

        if normalized_conclusion in self.FAILURE_CONCLUSIONS:
            return "failed"

        if normalized_conclusion in self.SUCCESS_CONCLUSIONS:
            return "succeeded"

        return "pending"

    def _calculate_overall_status(
        self,
        checks: list[CICheckResult],
    ) -> str:

        if not checks:
            return "no_checks"

        if any(check.state == "failed" for check in checks):
            return "failed"

        if any(check.state == "pending" for check in checks):
            return "pending"

        return "succeeded"

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
