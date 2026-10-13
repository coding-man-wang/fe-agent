from typing import Any

import pytest

from agent_core.git.github_review_feedback_provider import (
    GitHubReviewFeedbackError,
    GitHubReviewFeedbackProvider,
)
from agent_core.git.pull_request_model import (
    PullRequestResult,
)


class FakeGitHubReviewFeedbackApiClient:

    def __init__(
        self,
        *,
        review_comments: list[dict[str, Any]],
        conversation_comments: list[dict[str, Any]],
    ) -> None:

        self.review_comments = review_comments

        self.conversation_comments = conversation_comments

    async def get_review_comments(
        self,
        *,
        repository: str,
        pull_request_number: int,
    ) -> list[dict[str, Any]]:

        return self.review_comments

    async def get_conversation_comments(
        self,
        *,
        repository: str,
        pull_request_number: int,
    ) -> list[dict[str, Any]]:

        return self.conversation_comments


def create_pull_request() -> PullRequestResult:

    return PullRequestResult(
        handoff_id="a" * 64,
        task_id="TASK-001",
        proposal_id="b" * 64,
        provider="github",
        repository=("coding-man-wang/" "fe-agent"),
        number=123,
        url=("https://github.com/" "coding-man-wang/" "fe-agent/pull/123"),
        head_branch=("fe-agent/TASK-001"),
        base_branch="main",
        commit_sha="c" * 40,
    )


@pytest.mark.asyncio
async def test_collect_review_feedback():

    api_client = FakeGitHubReviewFeedbackApiClient(
        review_comments=[
            {
                "id": 101,
                "body": ("请复用 requestClient"),
                "path": ("src/user.ts"),
                "line": 42,
                "commit_id": ("c" * 40),
                "html_url": ("https://github.com/" "comment/101"),
                "user": {
                    "login": "alice",
                },
            }
        ],
        conversation_comments=[
            {
                "id": 202,
                "body": ("整体建议补充异常处理"),
                "html_url": ("https://github.com/" "comment/202"),
                "user": {
                    "login": "bob",
                },
            }
        ],
    )

    provider = GitHubReviewFeedbackProvider(api_client=api_client)

    result = await provider.collect(pull_request=(create_pull_request()))

    assert len(result) == 2

    inline = next(item for item in result if item.kind == "inline_comment")

    assert inline.reviewer == "alice"

    assert inline.relative_path == "src/user.ts"

    assert inline.line == 42

    assert inline.commit_sha == "c" * 40

    conversation = next(item for item in result if item.kind == "conversation_comment")

    assert conversation.reviewer == "bob"

    assert conversation.relative_path is None


@pytest.mark.asyncio
async def test_collect_ignores_empty_body():

    api_client = FakeGitHubReviewFeedbackApiClient(
        review_comments=[
            {
                "id": 101,
                "body": "   ",
                "path": ("src/user.ts"),
                "line": 42,
                "user": {"login": "alice"},
            }
        ],
        conversation_comments=[],
    )

    provider = GitHubReviewFeedbackProvider(api_client=api_client)

    result = await provider.collect(pull_request=(create_pull_request()))

    assert result == []


@pytest.mark.asyncio
async def test_collect_rejects_invalid_inline_id():

    api_client = FakeGitHubReviewFeedbackApiClient(
        review_comments=[
            {
                "id": "bad",
                "body": "test",
                "path": "src/user.ts",
                "user": {"login": "alice"},
            }
        ],
        conversation_comments=[],
    )

    provider = GitHubReviewFeedbackProvider(api_client=api_client)

    with pytest.raises(
        GitHubReviewFeedbackError,
        match="id 非法",
    ):
        await provider.collect(pull_request=(create_pull_request()))


@pytest.mark.asyncio
async def test_collect_rejects_non_github_pr():

    pull_request = create_pull_request().model_copy(update={"provider": "gitlab"})

    provider = GitHubReviewFeedbackProvider(
        api_client=(
            FakeGitHubReviewFeedbackApiClient(
                review_comments=[],
                conversation_comments=[],
            )
        )
    )

    with pytest.raises(
        GitHubReviewFeedbackError,
        match=("不是 GitHub Provider"),
    ):
        await provider.collect(pull_request=pull_request)
