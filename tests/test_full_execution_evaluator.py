from __future__ import annotations

from agent_core.benchmark.case_model import (
    BenchmarkCase,
)
from agent_core.benchmark.execution_evaluator import (
    FullExecutionEvaluator,
)
from agent_core.benchmark.execution_model import (
    FullExecutionTaskRecord,
)
from agent_core.benchmark.impact_verification import (
    ImpactVerificationPlan,
)
from agent_core.codebase.impact import (
    ChangeImpactReport,
    ImpactedFile,
)
from agent_core.coding.verification_model import (
    PatchVerificationResult,
)

# ============================================================
# Test Data Builders
# ============================================================


def _hidden_acceptance_passed(
    *,
    task_id: str = "CASE-TEST",
) -> PatchVerificationResult:
    """
    构造一个通过的 Hidden Acceptance。

    FullExecutionEvaluator 要求：
        hidden_acceptance.status == passed

    否则整个 Evaluation 会因为
    hidden_acceptance_failed 而失败。
    """

    return PatchVerificationResult(
        proposal_id="a" * 64,
        task_id=task_id,
        status="passed",
        checks=[],
    )


def _make_case(
    *,
    required_impact_files: list[str] | None = None,
    required_impact_verification_command_ids: list[str] | None = None,
) -> BenchmarkCase:
    """
    构造 Schema v2 Benchmark Case。

    这里故意保持 Change Oracle 很简单：

        expected_change_files:
            src/stores/users.ts

    这样测试重点只放在：

        Impact Analysis
        +
        Impact Verification
    """

    return BenchmarkCase.model_validate(
        {
            "schema_version": 2,
            "case_id": "CASE-TEST",
            "title": "Impact evaluator test",
            "category": "cross_file_change",
            "difficulty": "medium",
            "repository_name": ("fe-agent-benchmark-web"),
            "base_commit_sha": "1" * 40,
            "requirement": ("修改用户 Store，" "并验证相关调用方。"),
            "expectation": {
                # Schema v2 必填。
                "required_context_files": ["src/stores/users.ts"],
                "useful_context_files": [],
                "forbidden_context_files": [],
                # Change Oracle
                "expected_change_files": ["src/stores/users.ts"],
                "allowed_change_files": [],
                "forbidden_change_files": [],
                # Requirement
                "expected_behaviors": ["用户 Store 行为正确"],
                # Case 原始验证命令
                "verification_command_ids": [
                    "frontend-lint",
                    "frontend-unit-test",
                ],
                # v1.2.4 Impact Oracle
                "required_impact_files": (required_impact_files or []),
                ("required_impact_" "verification_command_ids"): (
                    required_impact_verification_command_ids or []
                ),
            },
        }
    )


def _production_impact(
    *,
    relative_path: str,
    source_changed_file: str = ("src/stores/users.ts"),
    depth: int = 1,
) -> ImpactedFile:
    """
    构造一个生产代码影响记录。
    """

    return ImpactedFile(
        relative_path=relative_path,
        kind="production",
        category="source",
        depth=depth,
        source_changed_file=(source_changed_file),
        via_path=(source_changed_file if depth == 1 else "src/views/UsersView.vue"),
        reason=(f"{relative_path} " f"受到 {source_changed_file} 影响"),
    )


def _test_impact(
    *,
    relative_path: str,
    source_changed_file: str = ("src/stores/users.ts"),
    depth: int = 1,
) -> ImpactedFile:
    """
    构造一个测试文件影响记录。
    """

    return ImpactedFile(
        relative_path=relative_path,
        kind="test",
        category="test",
        depth=depth,
        source_changed_file=(source_changed_file),
        via_path=source_changed_file,
        reason=(f"{relative_path} " f"直接验证 {source_changed_file}"),
    )


def _make_impact_report(
    *,
    production_files: list[str] | None = None,
    test_files: list[str] | None = None,
    suggested_test_files: list[str] | None = None,
) -> ChangeImpactReport:
    """
    构造 Impact Analysis 结果。
    """

    production_files = production_files or []

    test_files = test_files or []

    return ChangeImpactReport(
        repository_revision="2" * 40,
        changed_files=["src/stores/users.ts"],
        affected_production_files=[
            _production_impact(relative_path=path) for path in production_files
        ],
        affected_test_files=[_test_impact(relative_path=path) for path in test_files],
        suggested_test_files=(suggested_test_files or []),
        unresolved_changed_files=[],
        impact_level=("medium" if (production_files or test_files) else "low"),
        reasons=["test impact report"],
    )


def _make_verification_plan(
    *,
    final_command_ids: list[str],
) -> ImpactVerificationPlan:
    """
    构造 Impact Analysis 之后的
    Verification 决策。
    """

    base = [
        "frontend-lint",
        "frontend-unit-test",
    ]

    added = [command_id for command_id in final_command_ids if command_id not in base]

    return ImpactVerificationPlan(
        base_command_ids=base,
        added_command_ids=added,
        final_command_ids=(final_command_ids),
        reasons=["根据 Impact Analysis " "补充 Verification"] if added else [],
    )


def _completed_task(
    *,
    impact: ChangeImpactReport | None,
    verification: ImpactVerificationPlan | None,
) -> FullExecutionTaskRecord:
    """
    构造一个已经成功完成的 code_change Task。

    Evaluation 中 verification_passed
    依赖：

        status == completed
        verification_status == succeeded
    """

    return FullExecutionTaskRecord(
        task_id="TASK-002",
        task_type="code_change",
        status="completed",
        execution_disposition=("change_required"),
        base_commit_sha="1" * 40,
        context_revision="1" * 40,
        context_retrieval_mode="hybrid",
        semantic_fallback_reason=None,
        retrieval_files=[
            "src/stores/users.ts",
            "src/views/UsersView.vue",
        ],
        selected_context_files=[
            "src/stores/users.ts",
            "src/views/UsersView.vue",
        ],
        graph_added_files=[],
        workspace_id="workspace-test",
        proposal_id="b" * 64,
        proposed_files=["src/stores/users.ts"],
        planned_impact=impact,
        planned_verification=(verification),
        verification_status="succeeded",
        verification_checks=[],
        scheduled_verification_command_ids=[
            "frontend-unit-test",
        ],
        executed_verification_command_ids=[
            "frontend-unit-test",
        ],
        completion_status="completed",
        commit_sha="2" * 40,
        error=None,
    )


# ============================================================
# Backward Compatibility
# ============================================================


def test_evaluator_keeps_old_case_behavior_when_no_impact_oracle():
    """
    没有配置 Impact Oracle 时：

        Impact Gate 不应该影响旧 Benchmark。

    这是非常重要的兼容性测试。
    """

    case = _make_case()

    task = _completed_task(
        impact=None,
        verification=None,
    )

    evaluation = FullExecutionEvaluator().evaluate(
        case=case,
        actual_change_files=["src/stores/users.ts"],
        task_records=[task],
        hidden_acceptance=(_hidden_acceptance_passed()),
    )

    assert evaluation.status == "passed"

    assert evaluation.expected_impact_files == []

    assert evaluation.missed_impact_files == []

    assert evaluation.required_impact_verification_command_ids == []

    assert evaluation.missing_impact_verification_command_ids == []

    assert evaluation.impact_coverage_passed is True

    assert evaluation.impact_verification_passed is True

    assert evaluation.failure_reasons == []


# ============================================================
# Impact Coverage PASS
# ============================================================


def test_evaluator_passes_when_required_impact_and_verification_are_covered():
    """
    Oracle 要求：

        UsersView.vue
        usersStore.test.ts

    实际 Impact Analysis 都发现。

    同时要求：

        frontend-e2e

    Verification Plan 也包含。

    最终应该 PASS。
    """

    case = _make_case(
        required_impact_files=[
            "src/views/UsersView.vue",
            ("tests/unit/" "usersStore.test.ts"),
        ],
        required_impact_verification_command_ids=["frontend-e2e"],
    )

    impact = _make_impact_report(
        production_files=[
            "src/views/UsersView.vue",
            "src/router/index.ts",
        ],
        test_files=["tests/unit/" "usersStore.test.ts"],
        suggested_test_files=[
            ("tests/unit/" "usersStore.test.ts"),
            "tests/e2e/users.spec.ts",
        ],
    )

    verification = _make_verification_plan(
        final_command_ids=[
            "frontend-lint",
            "frontend-unit-test",
            "frontend-e2e",
        ]
    )

    task = _completed_task(
        impact=impact,
        verification=verification,
    )

    evaluation = FullExecutionEvaluator().evaluate(
        case=case,
        actual_change_files=["src/stores/users.ts"],
        task_records=[task],
        hidden_acceptance=(_hidden_acceptance_passed()),
    )

    assert evaluation.status == "passed"

    assert evaluation.expected_impact_files == [
        "src/views/UsersView.vue",
        ("tests/unit/" "usersStore.test.ts"),
    ]

    # 实际可以比 Oracle 多。
    #
    # 例如 router/index.ts
    # 也是真实合理的影响。
    assert "src/views/UsersView.vue" in evaluation.observed_impact_files

    assert "src/router/index.ts" in evaluation.observed_impact_files

    assert ("tests/unit/" "usersStore.test.ts") in evaluation.observed_impact_files

    assert evaluation.missed_impact_files == []

    assert evaluation.impact_coverage_passed is True

    assert evaluation.required_impact_verification_command_ids == ["frontend-e2e"]

    assert "frontend-e2e" in (evaluation.observed_verification_command_ids)

    assert evaluation.missing_impact_verification_command_ids == []

    assert evaluation.impact_verification_passed is True

    assert evaluation.failure_reasons == []


# ============================================================
# Missing Impact
# ============================================================


def test_evaluator_fails_when_required_impact_is_missing():
    """
    Oracle 明确要求：

        UsersView.vue

    但 Impact Analysis 没发现。

    即使：
        - Patch 正确
        - Verification succeeded
        - Hidden Acceptance passed

    Benchmark 也必须失败。

    这才说明 Impact Analysis
    真正成为 Benchmark Gate。
    """

    case = _make_case(
        required_impact_files=["src/views/UsersView.vue"],
    )

    impact = _make_impact_report(
        production_files=[],
        test_files=[],
        suggested_test_files=[],
    )

    verification = _make_verification_plan(
        final_command_ids=[
            "frontend-lint",
            "frontend-unit-test",
        ]
    )

    task = _completed_task(
        impact=impact,
        verification=verification,
    )

    evaluation = FullExecutionEvaluator().evaluate(
        case=case,
        actual_change_files=["src/stores/users.ts"],
        task_records=[task],
        hidden_acceptance=(_hidden_acceptance_passed()),
    )

    assert evaluation.status == "failed"

    assert evaluation.missed_impact_files == ["src/views/UsersView.vue"]

    assert evaluation.impact_coverage_passed is False

    assert "missing_required_impact" in evaluation.failure_reasons

    # Change 本身仍然是正确的。
    assert evaluation.missed_change_files == []

    assert evaluation.unexpected_change_files == []

    # Runtime Verification 本身也通过。
    #
    # Benchmark 失败的原因应该明确来自
    # Impact Gate。
    assert evaluation.verification_passed is True

    assert evaluation.hidden_acceptance_passed is True


# ============================================================
# Missing Impact Verification
# ============================================================


def test_evaluator_fails_when_required_impact_verification_is_missing():
    """
    Impact Analysis 找对了文件，

    但是 Oracle 要求：

        frontend-e2e

    最终 Verification Plan 没有它。

    Benchmark 必须失败。
    """

    case = _make_case(
        required_impact_files=["src/views/UsersView.vue"],
        required_impact_verification_command_ids=["frontend-e2e"],
    )

    impact = _make_impact_report(
        production_files=["src/views/UsersView.vue"],
        test_files=[],
        suggested_test_files=["tests/e2e/users.spec.ts"],
    )

    # 故意不包含 frontend-e2e
    verification = _make_verification_plan(
        final_command_ids=[
            "frontend-lint",
            "frontend-unit-test",
        ]
    )

    task = _completed_task(
        impact=impact,
        verification=verification,
    )

    evaluation = FullExecutionEvaluator().evaluate(
        case=case,
        actual_change_files=["src/stores/users.ts"],
        task_records=[task],
        hidden_acceptance=(_hidden_acceptance_passed()),
    )

    assert evaluation.status == "failed"

    # Impact 文件本身是找对的。
    assert evaluation.missed_impact_files == []

    assert evaluation.impact_coverage_passed is True

    # 真正失败的是 Verification Coverage。
    assert evaluation.missing_impact_verification_command_ids == ["frontend-e2e"]

    assert evaluation.impact_verification_passed is False

    assert "missing_impact_verification" in evaluation.failure_reasons

    # 原本 Runtime Verification / Hidden
    # Acceptance 都可以成功。
    #
    # 现在失败的是 v1.2.4 新增的
    # Benchmark Gate。
    assert evaluation.verification_passed is True

    assert evaluation.hidden_acceptance_passed is True


def _final_verification_task(
    *,
    executed_commands: list[str],
) -> FullExecutionTaskRecord:

    return FullExecutionTaskRecord(
        task_id="TASK-999",
        task_type="verification",
        status="completed",
        base_commit_sha="2" * 40,
        context_revision="2" * 40,
        scheduled_verification_command_ids=(list(executed_commands)),
        executed_verification_command_ids=(list(executed_commands)),
        verification_status=("succeeded"),
        verification_checks=[],
    )
