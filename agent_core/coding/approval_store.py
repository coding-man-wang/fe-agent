import re
from pathlib import Path

from agent_core.coding.approval import (
    ApprovalDecision,
    ApprovalError,
    ApprovalService,
)
from agent_core.coding.handoff import (
    write_json_atomically,
)
from agent_core.coding.review_model import (
    ApprovalReceipt,
    ReviewPackage,
)


class ApprovalStore:

    def __init__(
        self,
        repository_root: str | Path,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        self.directory = self.root / ".fe-agent" / "runtime" / "approvals"

    def _receipt_path(
        self,
        proposal_id: str,
    ) -> Path:

        if not re.fullmatch(
            r"[0-9a-f]{64}",
            proposal_id,
        ):
            raise ApprovalError("非法 Proposal ID")

        return self.directory / f"{proposal_id}.json"

    def record_decision(
        self,
        *,
        review: ReviewPackage,
        decision: ApprovalDecision,
        reason: str = "",
    ) -> ApprovalReceipt:

        path = self._receipt_path(review.proposal_id)

        if path.exists():
            raise ApprovalError("该 Proposal 已有持久化审批记录")

        service = ApprovalService()

        receipt = service.decide(
            review=review,
            decision=decision,
            reason=reason,
        )

        write_json_atomically(
            path,
            receipt.model_dump_json(indent=2),
        )

        return receipt

    def load(
        self,
        proposal_id: str,
    ) -> ApprovalReceipt:

        path = self._receipt_path(proposal_id)

        if not path.is_file():
            raise ApprovalError("找不到审批记录")

        try:
            receipt = ApprovalReceipt.model_validate_json(
                path.read_text(encoding="utf-8")
            )

        except (
            OSError,
            ValueError,
        ) as exc:
            raise ApprovalError("审批记录无法读取或格式非法") from exc

        if receipt.proposal_id != proposal_id:
            raise ApprovalError("审批记录的 Proposal ID 不一致")

        return receipt

    def require_approved(
        self,
        *,
        review: ReviewPackage,
    ) -> ApprovalReceipt:

        receipt = self.load(review.proposal_id)

        if receipt.task_id != review.task_id:
            raise ApprovalError("审批记录的 Task ID 不一致")

        if receipt.decision != "approved":
            raise ApprovalError("当前 Proposal 未获批准")

        return receipt
