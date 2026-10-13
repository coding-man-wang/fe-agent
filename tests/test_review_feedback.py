from pathlib import Path

from agent_core.git.pull_request_model import (
    PullRequestResult,
)
from agent_core.git.review_feedback import (
    ReviewFeedbackService,
)
from agent_core.git.review_feedback_model import (
    ReviewFeedbackItem,
)
from agent_core.git.review_feedback_store import (
    ReviewFeedbackStore,
)


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


def create_feedback(
    *,
    feedback_id: str,
    external_id: str,
) -> ReviewFeedbackItem:

    return ReviewFeedbackItem(
        feedback_id=feedback_id,
        kind="inline_comment",
        reviewer="alice",
        body=("请复用 requestClient"),
        relative_path=("src/user.ts"),
        line=42,
        commit_sha="c" * 40,
        external_id=external_id,
        url=None,
    )


def test_build_review_feedback_artifact():

    service = ReviewFeedbackService()

    item = create_feedback(
        feedback_id="d" * 64,
        external_id="101",
    )

    artifact = service.build_artifact(
        pull_request=(create_pull_request()),
        items=[item],
    )

    assert artifact.task_id == "TASK-001"

    assert artifact.commit_sha == "c" * 40

    assert len(artifact.items) == 1

    assert len(artifact.artifact_id) == 64


def test_artifact_is_deterministic():

    service = ReviewFeedbackService()

    item = create_feedback(
        feedback_id="d" * 64,
        external_id="101",
    )

    first = service.build_artifact(
        pull_request=(create_pull_request()),
        items=[item],
    )

    second = service.build_artifact(
        pull_request=(create_pull_request()),
        items=[item],
    )

    assert first.artifact_id == second.artifact_id


def test_duplicate_feedback_is_deduplicated():

    service = ReviewFeedbackService()

    item = create_feedback(
        feedback_id="d" * 64,
        external_id="101",
    )

    artifact = service.build_artifact(
        pull_request=(create_pull_request()),
        items=[
            item,
            item,
        ],
    )

    assert len(artifact.items) == 1


def test_store_round_trip(
    tmp_path: Path,
):

    repository = tmp_path / "repo"

    repository.mkdir()

    service = ReviewFeedbackService()

    item = create_feedback(
        feedback_id="d" * 64,
        external_id="101",
    )

    artifact = service.build_artifact(
        pull_request=(create_pull_request()),
        items=[item],
    )

    store = ReviewFeedbackStore(repository)

    store.save(artifact)

    loaded = store.load(artifact.artifact_id)

    assert loaded == artifact
