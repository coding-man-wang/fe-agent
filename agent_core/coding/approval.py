from typing import Literal

from agent_core.coding.review_model import (
    ApprovalReceipt,
    ReviewPackage,
)


class ApprovalError(ValueError):
    """审批状态或参数不合法。"""


ApprovalDecision = Literal[
    "approved",
    "rejected",
]


class ApprovalService:

    def __init__(self) -> None:

        self.receipts: dict[
            str,
            ApprovalReceipt,
        ] = {}

    def decide(
        self,
        *,
        review: ReviewPackage,
        decision: ApprovalDecision,
        reason: str = "",
    ) -> ApprovalReceipt:

        if decision not in {
            "approved",
            "rejected",
        }:
            raise ApprovalError("未知的审批决定")

        if decision == "rejected" and not reason.strip():
            raise ApprovalError("拒绝修改时必须填写原因")

        if review.proposal_id in self.receipts:
            raise ApprovalError("该 Proposal 已经存在审批决定")

        receipt = ApprovalReceipt(
            proposal_id=review.proposal_id,
            task_id=review.task_id,
            decision=decision,
            reason=reason.strip(),
        )

        self.receipts[review.proposal_id] = receipt

        return receipt

    def require_approved(
        self,
        *,
        review: ReviewPackage,
        receipt: ApprovalReceipt,
    ) -> None:

        if receipt.decision != "approved":
            raise ApprovalError("当前 Proposal 未获批准")

        if receipt.proposal_id != review.proposal_id:
            raise ApprovalError("审批凭证不属于当前 Proposal")

        if receipt.task_id != review.task_id:
            raise ApprovalError("审批凭证的 Task ID 不一致")

        recorded = self.receipts.get(receipt.proposal_id)

        if recorded != receipt:
            raise ApprovalError("审批凭证未被当前服务记录")
