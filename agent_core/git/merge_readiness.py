from agent_core.git.ci_tracking_model import (
    CITrackingResult,
)
from agent_core.git.merge_readiness_model import (
    MergeReadinessResult,
)
from agent_core.git.review_model import (
    PullRequestReviewStatus,
)


class MergeReadinessGate:

    def evaluate(
        self,
        *,
        ci: CITrackingResult,
        review: PullRequestReviewStatus,
    ) -> MergeReadinessResult:

        reasons: list[str] = []

        # --------------------------
        # CI Gate
        # --------------------------

        if ci.outcome != "succeeded":

            reasons.append(f"CI 未通过: {ci.outcome}")

            return MergeReadinessResult(
                status="blocked_by_ci",
                reasons=reasons,
                ci_succeeded=False,
                review_approved=(review.status == "approved"),
            )

        # --------------------------
        # Review Gate
        # --------------------------

        if review.status == "changes_requested":

            reasons.append("Code Review 存在 " "Changes Requested")

            return MergeReadinessResult(
                status="blocked",
                reasons=reasons,
                ci_succeeded=True,
                review_approved=False,
            )

        if review.status != "approved":

            reasons.append("等待 Code Review Approval")

            return MergeReadinessResult(
                status="waiting_for_review",
                reasons=reasons,
                ci_succeeded=True,
                review_approved=False,
            )

        return MergeReadinessResult(
            status=("ready_for_human_merge"),
            reasons=[],
            ci_succeeded=True,
            review_approved=True,
        )
