import sys
from pathlib import Path

import pytest

from agent_core.coding.approval_store import (
    ApprovalStore,
)
from agent_core.coding.execution_journal_model import (
    FileExecutionRecord,
    PatchExecutionJournal,
)
from agent_core.coding.reviewer import (
    PatchReviewer,
)
from agent_core.coding.verification_command_model import (
    AllowedVerificationCommand,
)
from agent_core.coding.verification_executor import (
    VerificationExecutor,
)
from agent_core.coding.verification_plan import (
    VerificationPlan,
)
from agent_core.coding.verification_policy import (
    VerificationCommandPolicy,
)
from agent_core.coding.verification_store import (
    VerificationStore,
)
from agent_core.coding.verification_workflow import (
    VerificationWorkflow,
    VerificationWorkflowError,
)
from tests.test_recovery import (
    AFTER_A,
    AFTER_B,
    BEFORE_B,
    create_repository,
    make_proposal,
)


def create_applied_journal(
    proposal,
):
    review = PatchReviewer().prepare(proposal)

    return PatchExecutionJournal(
        proposal_id=(review.proposal_id),
        task_id=(proposal.task_id),
        status="applied",
        files=[
            FileExecutionRecord(
                relative_path=(patch.relative_path),
                expected_sha256=(patch.expected_sha256),
                after_sha256=(patch.after_sha256),
                status="applied",
            )
            for patch in proposal.patches
        ],
    )


def create_policy(
    *,
    should_pass: bool,
):
    code = "import sys; " "print('ok'); " f"sys.exit({0 if should_pass else 2})"

    return VerificationCommandPolicy(
        commands=[
            AllowedVerificationCommand(
                command_id="verify",
                name="Verification",
                argv=[
                    sys.executable,
                    "-c",
                    code,
                ],
                timeout_seconds=10,
            )
        ]
    )


def create_workflow(
    root: Path,
    *,
    should_pass: bool,
):
    approval_store = ApprovalStore(root)

    verification_store = VerificationStore(root)

    executor = VerificationExecutor(
        repository_root=root,
        policy=create_policy(should_pass=(should_pass)),
    )

    workflow = VerificationWorkflow(
        repository_root=root,
        approval_store=(approval_store),
        verification_store=(verification_store),
        executor=executor,
    )

    return (
        approval_store,
        verification_store,
        workflow,
    )


@pytest.mark.asyncio
async def test_verification_workflow_passes(
    tmp_path: Path,
):
    proposal = make_proposal()

    create_repository(
        tmp_path,
        content_a=AFTER_A,
        content_b=AFTER_B,
    )

    journal = create_applied_journal(proposal)

    (
        approval_store,
        verification_store,
        workflow,
    ) = create_workflow(
        tmp_path,
        should_pass=True,
    )

    review = PatchReviewer().prepare(proposal)

    approval_store.record_decision(
        review=review,
        decision="approved",
    )

    plan = VerificationPlan(
        proposal_id=(review.proposal_id),
        task_id=(proposal.task_id),
        command_ids=[
            "verify",
        ],
    )

    record = await workflow.run(
        proposal=proposal,
        journal=journal,
        plan=plan,
    )

    assert record.status == "succeeded"

    assert record.attempts == 1

    assert record.result is not None

    assert record.result.status == "passed"

    persisted = verification_store.load(review.proposal_id)

    assert persisted.status == "succeeded"


@pytest.mark.asyncio
async def test_verification_workflow_records_failure(
    tmp_path: Path,
):
    proposal = make_proposal()

    create_repository(
        tmp_path,
        content_a=AFTER_A,
        content_b=AFTER_B,
    )

    journal = create_applied_journal(proposal)

    (
        approval_store,
        _,
        workflow,
    ) = create_workflow(
        tmp_path,
        should_pass=False,
    )

    review = PatchReviewer().prepare(proposal)

    approval_store.record_decision(
        review=review,
        decision="approved",
    )

    record = await workflow.run(
        proposal=proposal,
        journal=journal,
        plan=VerificationPlan(
            proposal_id=(review.proposal_id),
            task_id=(proposal.task_id),
            command_ids=[
                "verify",
            ],
        ),
    )

    assert record.status == "failed"

    assert record.result is not None

    assert record.result.status == "failed"


@pytest.mark.asyncio
async def test_verification_requires_applied_journal(
    tmp_path: Path,
):
    proposal = make_proposal()

    create_repository(
        tmp_path,
        content_a=AFTER_A,
        content_b=BEFORE_B,
    )

    journal = create_applied_journal(proposal)

    journal.status = "partially_applied"

    journal.files[1].status = "pending"

    (
        approval_store,
        _,
        workflow,
    ) = create_workflow(
        tmp_path,
        should_pass=True,
    )

    review = PatchReviewer().prepare(proposal)

    approval_store.record_decision(
        review=review,
        decision="approved",
    )

    with pytest.raises(
        VerificationWorkflowError,
        match="applied",
    ):
        await workflow.run(
            proposal=proposal,
            journal=journal,
            plan=VerificationPlan(
                proposal_id=(review.proposal_id),
                task_id=(proposal.task_id),
                command_ids=[
                    "verify",
                ],
            ),
        )


@pytest.mark.asyncio
async def test_verification_rejects_diverged_repository(
    tmp_path: Path,
):
    proposal = make_proposal()

    path_a, _ = create_repository(
        tmp_path,
        content_a=AFTER_A,
        content_b=AFTER_B,
    )

    journal = create_applied_journal(proposal)

    (
        approval_store,
        _,
        workflow,
    ) = create_workflow(
        tmp_path,
        should_pass=True,
    )

    review = PatchReviewer().prepare(proposal)

    approval_store.record_decision(
        review=review,
        decision="approved",
    )

    path_a.write_bytes(b"external change\n")

    with pytest.raises(
        VerificationWorkflowError,
        match="仓库状态",
    ):
        await workflow.run(
            proposal=proposal,
            journal=journal,
            plan=VerificationPlan(
                proposal_id=(review.proposal_id),
                task_id=(proposal.task_id),
                command_ids=[
                    "verify",
                ],
            ),
        )


@pytest.mark.asyncio
async def test_successful_verification_is_reused(
    tmp_path: Path,
):
    proposal = make_proposal()

    create_repository(
        tmp_path,
        content_a=AFTER_A,
        content_b=AFTER_B,
    )

    journal = create_applied_journal(proposal)

    (
        approval_store,
        _,
        workflow,
    ) = create_workflow(
        tmp_path,
        should_pass=True,
    )

    review = PatchReviewer().prepare(proposal)

    approval_store.record_decision(
        review=review,
        decision="approved",
    )

    plan = VerificationPlan(
        proposal_id=(review.proposal_id),
        task_id=(proposal.task_id),
        command_ids=[
            "verify",
        ],
    )

    first = await workflow.run(
        proposal=proposal,
        journal=journal,
        plan=plan,
    )

    second = await workflow.run(
        proposal=proposal,
        journal=journal,
        plan=plan,
    )

    assert second.status == "succeeded"

    # 没有产生新 Attempt。
    assert second.attempts == first.attempts == 1
