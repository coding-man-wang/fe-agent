from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from agent_core.benchmark.case_model import BenchmarkCase
from agent_core.benchmark.context_p3_cases import build_cases

from apps.agent_cli.benchmark_active_context_v1_4_9 import (
    validate_canary_report,
)

# D2-A 的固定实验范围。
# 不从 LLM / Shadow Report / Benchmark Oracle 中推导权限。
EXPERIMENT_CASE = "CASE-006"
EXPERIMENT_TASK = "TASK-002"

REVIEWED_TEST_FILE = "tests/unit/UsersView.test.ts"

SHA40 = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ActivePreflightDecision:
    version: str
    status: str
    reason: str
    case_id: str
    task_id: str | None
    approved_revision: str | None
    current_revision: str
    canary_sha256: str | None

    # D2-A 只检查准入，绝不执行 Active。
    applied: bool = False
    production_ready: bool = False
    active_execution_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_active_preflight(
    *,
    case: BenchmarkCase,
    task_id: str | None,
    approved_revision: str | None,
    current_revision: str,
    repository_dirty: bool,
    canary_report: Mapping[str, Any] | None,
    canary_sha256: str | None,
) -> ActivePreflightDecision:
    """
    P4-B-D2-A: Controlled Active Preflight。

    这是前置检查，不是 Active 授权凭据。

    当前只能针对:
        CASE-006 / TASK-002 / 初始 Revision

    不接触:
        Retrieval
        LLM
        Context Bundle
        Patch
        Worktree
    """

    def decision(
        status: str,
        reason: str,
    ) -> ActivePreflightDecision:
        return ActivePreflightDecision(
            version="v1.4.9-P4-B-D2-A",
            status=status,
            reason=reason,
            case_id=case.case_id,
            task_id=task_id,
            approved_revision=approved_revision,
            current_revision=current_revision,
            canary_sha256=canary_sha256,
        )

    # 1. 固定实验对象。
    if case.case_id != EXPERIMENT_CASE:
        return decision(
            "denied",
            "case_not_allowlisted",
        )

    if task_id != EXPERIMENT_TASK:
        return decision(
            "denied",
            "task_not_allowlisted",
        )

    # 2. 必须使用已经人工审核的 Case Oracle。
    if REVIEWED_TEST_FILE not in (case.expectation.allowed_change_files):
        return decision(
            "denied",
            "case_oracle_not_reviewed",
        )

    # 3. 批准版本必须是明确的 Git SHA。
    if not isinstance(approved_revision, str) or not SHA40.fullmatch(approved_revision):
        return decision(
            "denied",
            "invalid_approved_revision",
        )

    if case.base_commit_sha != approved_revision:
        return decision(
            "denied",
            "case_base_revision_mismatch",
        )

    if current_revision != approved_revision:
        return decision(
            "denied",
            "repository_revision_mismatch",
        )

    # 4. 未提交的受跟踪文件变化不允许通过。
    if repository_dirty:
        return decision(
            "denied",
            "repository_dirty",
        )

    # 5. Canary 必须存在。
    if (
        not isinstance(canary_report, Mapping)
        or not isinstance(canary_sha256, str)
        or not canary_sha256
    ):
        return decision(
            "denied",
            "canary_unavailable",
        )

    # 6. 复用已有 P2-C 的完整门禁。
    expected_ids = {item.case_id for item in build_cases()}

    if not validate_canary_report(
        dict(canary_report),
        revision=approved_revision,
        expected_ids=expected_ids,
    ):
        return decision(
            "denied",
            "canary_invalid",
        )

    return decision(
        "passed",
        "active_preflight_valid",
    )
