from agent_core.git.review_model import (
    PullRequestReviewStatus,
)


def create_review(
    status: str,
) -> PullRequestReviewStatus:

    if status == "approved":
        approvals = 1
        changes_requested = 0

    elif status == "changes_requested":
        approvals = 0
        changes_requested = 1

    else:
        approvals = 0
        changes_requested = 0

    return PullRequestReviewStatus(
        provider="github",
        repository=("coding-man-wang/fe-agent"),
        pull_request_number=1,
        commit_sha="c" * 40,
        status=status,
        approvals=approvals,
        changes_requested=(changes_requested),
        reviews=[],
    )
