from typing import Any

import pytest

from agent_core.git.github_ci_provider import (
    GitHubCIError,
    GitHubCIProvider,
)
from agent_core.git.pull_request_model import (
    PullRequestResult,
)


class FakeGitHubCIApiClient:

    def __init__(
        self,
        *,
        combined_status: dict[str, Any] | None = None,
        check_runs: dict[str, Any] | None = None,
    ) -> None:

        self.combined_status = (
            combined_status
            if combined_status is not None
            else {
                "state": "pending",
                "statuses": [],
            }
        )

        self.check_runs = (
            check_runs
            if check_runs is not None
            else {
                "total_count": 0,
                "check_runs": [],
            }
        )

        self.status_calls: list[tuple[str, str]] = []

        self.check_calls: list[tuple[str, str]] = []

    async def get_combined_status(
        self,
        *,
        repository: str,
        commit_sha: str,
    ) -> dict[str, Any]:

        self.status_calls.append(
            (
                repository,
                commit_sha,
            )
        )

        return self.combined_status

    async def get_check_runs(
        self,
        *,
        repository: str,
        commit_sha: str,
    ) -> dict[str, Any]:

        self.check_calls.append(
            (
                repository,
                commit_sha,
            )
        )

        return self.check_runs


def create_pull_request() -> PullRequestResult:

    return PullRequestResult(
        handoff_id="a" * 64,
        task_id="TASK-001",
        proposal_id="b" * 64,
        provider="github",
        repository=("coding-man-wang/" "fe-agent"),
        number=123,
        url=("https://github.com/" "coding-man-wang/" "fe-agent/pull/123"),
        head_branch=("fe-agent/TASK-001-test"),
        base_branch="main",
        commit_sha="c" * 40,
    )


@pytest.mark.asyncio
async def test_ci_status_succeeded():
    pull_request = create_pull_request()

    api_client = FakeGitHubCIApiClient(
        combined_status={
            "state": "success",
            "statuses": [
                {
                    "context": ("external/build"),
                    "state": ("success"),
                    "target_url": ("https://ci.example/" "build"),
                }
            ],
        },
        check_runs={
            "total_count": 2,
            "check_runs": [
                {
                    "name": "pytest",
                    "status": ("completed"),
                    "conclusion": ("success"),
                    "details_url": ("https://github.com/" "checks/1"),
                },
                {
                    "name": "lint",
                    "status": ("completed"),
                    "conclusion": ("success"),
                    "details_url": ("https://github.com/" "checks/2"),
                },
            ],
        },
    )

    provider = GitHubCIProvider(api_client=api_client)

    result = await provider.get_status(pull_request=pull_request)

    assert result.status == "succeeded"

    assert result.repository == pull_request.repository

    assert result.pull_request_number == pull_request.number

    assert result.commit_sha == pull_request.commit_sha

    assert len(result.checks) == 3

    assert all(check.state == "succeeded" for check in result.checks)

    assert api_client.status_calls == [
        (
            pull_request.repository,
            pull_request.commit_sha,
        )
    ]

    assert api_client.check_calls == [
        (
            pull_request.repository,
            pull_request.commit_sha,
        )
    ]


@pytest.mark.asyncio
async def test_ci_status_pending_when_check_run_in_progress():
    pull_request = create_pull_request()

    api_client = FakeGitHubCIApiClient(
        combined_status={
            "state": "success",
            "statuses": [
                {
                    "context": "build",
                    "state": "success",
                }
            ],
        },
        check_runs={
            "total_count": 1,
            "check_runs": [
                {
                    "name": "e2e",
                    "status": ("in_progress"),
                    "conclusion": None,
                }
            ],
        },
    )

    provider = GitHubCIProvider(api_client=api_client)

    result = await provider.get_status(pull_request=pull_request)

    assert result.status == "pending"

    e2e = next(check for check in result.checks if check.name == "e2e")

    assert e2e.state == "pending"


@pytest.mark.asyncio
async def test_ci_status_failed_when_check_run_fails():
    pull_request = create_pull_request()

    api_client = FakeGitHubCIApiClient(
        combined_status={
            "state": "success",
            "statuses": [],
        },
        check_runs={
            "total_count": 2,
            "check_runs": [
                {
                    "name": "lint",
                    "status": ("completed"),
                    "conclusion": ("success"),
                },
                {
                    "name": "pytest",
                    "status": ("completed"),
                    "conclusion": ("failure"),
                },
            ],
        },
    )

    provider = GitHubCIProvider(api_client=api_client)

    result = await provider.get_status(pull_request=pull_request)

    assert result.status == "failed"

    pytest_check = next(check for check in result.checks if check.name == "pytest")

    assert pytest_check.state == "failed"


@pytest.mark.asyncio
async def test_ci_status_failed_when_commit_status_errors():
    pull_request = create_pull_request()

    api_client = FakeGitHubCIApiClient(
        combined_status={
            "state": "error",
            "statuses": [
                {
                    "context": ("security"),
                    "state": "error",
                }
            ],
        },
        check_runs={
            "total_count": 0,
            "check_runs": [],
        },
    )

    provider = GitHubCIProvider(api_client=api_client)

    result = await provider.get_status(pull_request=pull_request)

    assert result.status == "failed"


@pytest.mark.asyncio
async def test_ci_status_no_checks():
    pull_request = create_pull_request()

    api_client = FakeGitHubCIApiClient(
        combined_status={
            "state": "pending",
            "statuses": [],
        },
        check_runs={
            "total_count": 0,
            "check_runs": [],
        },
    )

    provider = GitHubCIProvider(api_client=api_client)

    result = await provider.get_status(pull_request=pull_request)

    assert result.status == "no_checks"

    assert result.checks == []


@pytest.mark.asyncio
async def test_ci_skipped_check_is_not_failure():
    pull_request = create_pull_request()

    api_client = FakeGitHubCIApiClient(
        check_runs={
            "total_count": 1,
            "check_runs": [
                {
                    "name": ("optional-e2e"),
                    "status": ("completed"),
                    "conclusion": ("skipped"),
                }
            ],
        },
    )

    provider = GitHubCIProvider(api_client=api_client)

    result = await provider.get_status(pull_request=pull_request)

    assert result.status == "succeeded"


@pytest.mark.asyncio
async def test_ci_cancelled_check_is_failure():
    pull_request = create_pull_request()

    api_client = FakeGitHubCIApiClient(
        check_runs={
            "total_count": 1,
            "check_runs": [
                {
                    "name": "build",
                    "status": ("completed"),
                    "conclusion": ("cancelled"),
                }
            ],
        },
    )

    provider = GitHubCIProvider(api_client=api_client)

    result = await provider.get_status(pull_request=pull_request)

    assert result.status == "failed"


@pytest.mark.asyncio
async def test_ci_rejects_non_github_pull_request():
    pull_request = create_pull_request().model_copy(update={"provider": "gitlab"})

    provider = GitHubCIProvider(api_client=(FakeGitHubCIApiClient()))

    with pytest.raises(
        GitHubCIError,
        match=("不是 GitHub Provider"),
    ):
        await provider.get_status(pull_request=pull_request)


@pytest.mark.asyncio
async def test_ci_rejects_invalid_statuses_shape():
    pull_request = create_pull_request()

    api_client = FakeGitHubCIApiClient(
        combined_status={
            "state": "success",
            "statuses": "invalid",
        }
    )

    provider = GitHubCIProvider(api_client=api_client)

    with pytest.raises(
        GitHubCIError,
        match="statuses 非法",
    ):
        await provider.get_status(pull_request=pull_request)


@pytest.mark.asyncio
async def test_ci_rejects_invalid_check_runs_shape():
    pull_request = create_pull_request()

    api_client = FakeGitHubCIApiClient(
        check_runs={
            "total_count": 1,
            "check_runs": ("invalid"),
        }
    )

    provider = GitHubCIProvider(api_client=api_client)

    with pytest.raises(
        GitHubCIError,
        match="check_runs 非法",
    ):
        await provider.get_status(pull_request=pull_request)
