import sys
from pathlib import Path

import pytest

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
    VerificationPolicyError,
)


def create_policy():
    return VerificationCommandPolicy(
        commands=[
            AllowedVerificationCommand(
                command_id="pass",
                name="Passing Check",
                argv=[
                    sys.executable,
                    "-c",
                    ("print(" "'verification passed'" ")"),
                ],
                timeout_seconds=10,
            ),
            AllowedVerificationCommand(
                command_id="fail",
                name="Failing Check",
                argv=[
                    sys.executable,
                    "-c",
                    ("import sys; " "print('verification failed'); " "sys.exit(2)"),
                ],
                timeout_seconds=10,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_verification_passes(
    tmp_path: Path,
):
    executor = VerificationExecutor(
        repository_root=tmp_path,
        policy=create_policy(),
    )

    result = await executor.run(
        plan=VerificationPlan(
            proposal_id="a" * 64,
            task_id="TASK-003",
            command_ids=[
                "pass",
            ],
        )
    )

    assert result.status == "passed"

    assert len(result.checks) == 1

    assert result.checks[0].status == "passed"


@pytest.mark.asyncio
async def test_verification_failure(
    tmp_path: Path,
):
    executor = VerificationExecutor(
        repository_root=tmp_path,
        policy=create_policy(),
    )

    result = await executor.run(
        plan=VerificationPlan(
            proposal_id="a" * 64,
            task_id="TASK-003",
            command_ids=[
                "fail",
            ],
        )
    )

    assert result.status == "failed"

    assert result.checks[0].status == "failed"

    assert "exit_code=2" in result.checks[0].details


@pytest.mark.asyncio
async def test_fail_fast(
    tmp_path: Path,
):
    executor = VerificationExecutor(
        repository_root=tmp_path,
        policy=create_policy(),
    )

    result = await executor.run(
        plan=VerificationPlan(
            proposal_id="a" * 64,
            task_id="TASK-003",
            command_ids=[
                "fail",
                "pass",
            ],
        )
    )

    # 第一个失败后，
    # 第二个不执行。
    assert len(result.checks) == 1

    assert result.status == "failed"


@pytest.mark.asyncio
async def test_unknown_command_rejected_before_execution(
    tmp_path: Path,
):
    executor = VerificationExecutor(
        repository_root=tmp_path,
        policy=create_policy(),
    )

    with pytest.raises(
        VerificationPolicyError,
        match="未授权",
    ):
        await executor.run(
            plan=VerificationPlan(
                proposal_id="a" * 64,
                task_id="TASK-003",
                command_ids=[
                    "pass",
                    "rm_everything",
                ],
            )
        )


def test_duplicate_policy_command_rejected():
    with pytest.raises(
        VerificationPolicyError,
        match="重复",
    ):
        VerificationCommandPolicy(
            commands=[
                AllowedVerificationCommand(
                    command_id="same",
                    name="A",
                    argv=[
                        sys.executable,
                        "-V",
                    ],
                ),
                AllowedVerificationCommand(
                    command_id="same",
                    name="B",
                    argv=[
                        sys.executable,
                        "-V",
                    ],
                ),
            ]
        )


@pytest.mark.asyncio
async def test_verification_timeout(
    tmp_path: Path,
):
    policy = VerificationCommandPolicy(
        commands=[
            AllowedVerificationCommand(
                command_id="slow",
                name="Slow Check",
                argv=[
                    sys.executable,
                    "-c",
                    ("import time; " "time.sleep(5)"),
                ],
                timeout_seconds=1,
            )
        ]
    )

    executor = VerificationExecutor(
        repository_root=tmp_path,
        policy=policy,
    )

    result = await executor.run(
        plan=VerificationPlan(
            proposal_id="a" * 64,
            task_id="TASK-003",
            command_ids=[
                "slow",
            ],
        )
    )

    assert result.status == "failed"

    assert "timed_out" in result.checks[0].details
