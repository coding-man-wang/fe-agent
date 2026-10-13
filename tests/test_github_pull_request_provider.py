from pathlib import Path
from typing import Any

import pytest

from agent_core.git.github_pull_request_provider import (
    GitHubPullRequestError,
    GitHubPullRequestProvider,
)
from agent_core.git.pull_request_model import (
    PullRequestRequest,
)


class FakeGitCommandResult:

    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
    ) -> None:

        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class FakeGitClient:

    def __init__(
        self,
        *,
        remote_url: str,
    ) -> None:

        self.remote_url = remote_url

        self.calls: list[
            tuple[
                Path,
                list[str],
                bool,
            ]
        ] = []

    async def run(
        self,
        *,
        cwd: Path,
        args: list[str],
        allow_failure: bool = False,
    ) -> FakeGitCommandResult:

        self.calls.append(
            (
                cwd,
                args,
                allow_failure,
            )
        )

        if args == [
            "remote",
            "get-url",
            "origin",
        ]:

            return FakeGitCommandResult(stdout=(self.remote_url + "\n"))

        raise AssertionError(f"Unexpected git args: {args}")


class FakeGitHubApiClient:

    def __init__(
        self,
        *,
        response: dict[str, Any],
    ) -> None:

        self.response = response

        self.calls: list[dict[str, Any]] = []

    async def create_pull_request(
        self,
        *,
        repository: str,
        title: str,
        body: str,
        head: str,
        base: str,
    ) -> dict[str, Any]:

        self.calls.append(
            {
                "repository": repository,
                "title": title,
                "body": body,
                "head": head,
                "base": base,
            }
        )

        return self.response


def create_request() -> PullRequestRequest:

    return PullRequestRequest(
        handoff_id="a" * 64,
        task_id="TASK-001",
        proposal_id="b" * 64,
        remote_name="origin",
        head_branch=("fe-agent/TASK-001-test"),
        base_branch="main",
        commit_sha="c" * 40,
        title=("feat: add user status filter"),
        body=("## Agent Change Summary"),
    )


def create_repository(
    tmp_path: Path,
) -> Path:

    repository = tmp_path / "repository"

    repository.mkdir()

    return repository


@pytest.mark.asyncio
async def test_create_github_pull_request(
    tmp_path: Path,
):
    repository_root = create_repository(tmp_path)

    request = create_request()

    git_client = FakeGitClient(
        remote_url=("https://github.com/" "coding-man-wang/" "fe-agent.git")
    )

    api_client = FakeGitHubApiClient(
        response={
            "number": 123,
            "html_url": ("https://github.com/" "coding-man-wang/" "fe-agent/pull/123"),
            "head": {
                "ref": (request.head_branch),
                "sha": (request.commit_sha),
            },
            "base": {
                "ref": (request.base_branch),
            },
        }
    )

    provider = GitHubPullRequestProvider(
        repository_root=(repository_root),
        git_client=git_client,
        api_client=api_client,
    )

    result = await provider.create(request=request)

    assert result.provider == ("github")

    assert result.repository == ("coding-man-wang/fe-agent")

    assert result.number == 123

    assert result.head_branch == request.head_branch

    assert result.base_branch == request.base_branch

    assert result.commit_sha == request.commit_sha

    assert len(api_client.calls) == 1

    call = api_client.calls[0]

    assert call["repository"] == ("coding-man-wang/fe-agent")

    assert call["head"] == (request.head_branch)

    assert call["base"] == (request.base_branch)


@pytest.mark.asyncio
async def test_create_supports_ssh_remote(
    tmp_path: Path,
):
    repository_root = create_repository(tmp_path)

    request = create_request()

    git_client = FakeGitClient(
        remote_url=("git@github.com:" "coding-man-wang/" "fe-agent.git")
    )

    api_client = FakeGitHubApiClient(
        response={
            "number": 12,
            "html_url": ("https://github.com/" "coding-man-wang/" "fe-agent/pull/12"),
            "head": {
                "ref": (request.head_branch),
                "sha": (request.commit_sha),
            },
            "base": {
                "ref": (request.base_branch),
            },
        }
    )

    provider = GitHubPullRequestProvider(
        repository_root=(repository_root),
        git_client=git_client,
        api_client=api_client,
    )

    result = await provider.create(request=request)

    assert result.repository == ("coding-man-wang/fe-agent")


@pytest.mark.asyncio
async def test_create_rejects_wrong_head_branch(
    tmp_path: Path,
):
    repository_root = create_repository(tmp_path)

    request = create_request()

    git_client = FakeGitClient(
        remote_url=("https://github.com/" "coding-man-wang/" "fe-agent.git")
    )

    api_client = FakeGitHubApiClient(
        response={
            "number": 20,
            "html_url": ("https://github.com/" "coding-man-wang/" "fe-agent/pull/20"),
            "head": {
                "ref": ("fe-agent/" "OTHER"),
                "sha": (request.commit_sha),
            },
            "base": {
                "ref": (request.base_branch),
            },
        }
    )

    provider = GitHubPullRequestProvider(
        repository_root=(repository_root),
        git_client=git_client,
        api_client=api_client,
    )

    with pytest.raises(
        GitHubPullRequestError,
        match=("Head Branch " "与请求不一致"),
    ):
        await provider.create(request=request)


@pytest.mark.asyncio
async def test_create_rejects_wrong_base_branch(
    tmp_path: Path,
):
    repository_root = create_repository(tmp_path)

    request = create_request()

    git_client = FakeGitClient(
        remote_url=("https://github.com/" "coding-man-wang/" "fe-agent.git")
    )

    api_client = FakeGitHubApiClient(
        response={
            "number": 21,
            "html_url": ("https://github.com/" "coding-man-wang/" "fe-agent/pull/21"),
            "head": {
                "ref": (request.head_branch),
                "sha": (request.commit_sha),
            },
            "base": {
                "ref": "develop",
            },
        }
    )

    provider = GitHubPullRequestProvider(
        repository_root=(repository_root),
        git_client=git_client,
        api_client=api_client,
    )

    with pytest.raises(
        GitHubPullRequestError,
        match=("Base Branch " "与请求不一致"),
    ):
        await provider.create(request=request)


@pytest.mark.asyncio
async def test_create_rejects_wrong_commit(
    tmp_path: Path,
):
    repository_root = create_repository(tmp_path)

    request = create_request()

    git_client = FakeGitClient(
        remote_url=("https://github.com/" "coding-man-wang/" "fe-agent.git")
    )

    api_client = FakeGitHubApiClient(
        response={
            "number": 22,
            "html_url": ("https://github.com/" "coding-man-wang/" "fe-agent/pull/22"),
            "head": {
                "ref": (request.head_branch),
                "sha": "d" * 40,
            },
            "base": {
                "ref": (request.base_branch),
            },
        }
    )

    provider = GitHubPullRequestProvider(
        repository_root=(repository_root),
        git_client=git_client,
        api_client=api_client,
    )

    with pytest.raises(
        GitHubPullRequestError,
        match=("Head Commit " "与请求不一致"),
    ):
        await provider.create(request=request)


@pytest.mark.asyncio
async def test_create_rejects_invalid_number(
    tmp_path: Path,
):
    repository_root = create_repository(tmp_path)

    request = create_request()

    git_client = FakeGitClient(
        remote_url=("https://github.com/" "coding-man-wang/" "fe-agent.git")
    )

    api_client = FakeGitHubApiClient(
        response={
            "number": 0,
            "html_url": "x",
            "head": {
                "ref": (request.head_branch),
                "sha": (request.commit_sha),
            },
            "base": {
                "ref": (request.base_branch),
            },
        }
    )

    provider = GitHubPullRequestProvider(
        repository_root=(repository_root),
        git_client=git_client,
        api_client=api_client,
    )

    with pytest.raises(
        GitHubPullRequestError,
        match="PR number 非法",
    ):
        await provider.create(request=request)


from agent_core.git.github_repository import (
    GitHubRepositoryError,
    parse_github_remote_url,
)


def test_parse_https_github_remote():

    result = parse_github_remote_url(
        "https://github.com/" "coding-man-wang/" "fe-agent.git"
    )

    assert result.owner == ("coding-man-wang")

    assert result.name == ("fe-agent")


def test_parse_ssh_github_remote():

    result = parse_github_remote_url(
        "git@github.com:" "coding-man-wang/" "fe-agent.git"
    )

    assert result.full_name == ("coding-man-wang/fe-agent")


def test_parse_rejects_non_github_remote():

    with pytest.raises(
        GitHubRepositoryError,
        match=("当前只支持 github.com"),
    ):
        parse_github_remote_url("https://gitlab.com/" "example/project.git")
