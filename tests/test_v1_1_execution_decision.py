from types import SimpleNamespace

from agent_core.benchmark.execution_decision import (
    RuntimeExecutionDecisionPolicy,
)
from agent_core.coding.model import (
    EvidenceGap,
    ProposedCodeChange,
    ReadOnlyCodingResult,
)


def make_result(
    *,
    changes=None,
    missing=None,
    gaps=None,
):
    return ReadOnlyCodingResult(
        task_id="TASK-004",
        summary="analysis",
        proposed_changes=changes or [],
        missing_information=missing or [],
        evidence_gaps=gaps or [],
    )


def make_surface():
    return SimpleNamespace(
        implementation_targets=[
            "src/stores/users.ts",
            "src/views/UsersView.vue",
        ],
        dependencies=[
            "src/api/users.ts",
        ],
        references=[
            "src/components/PaginationBar.vue",
            "tests/unit/usersStore.test.ts",
        ],
        tests=[],
    )


def test_runtime_policy_continues_when_change_is_proposed():
    decision = RuntimeExecutionDecisionPolicy().decide(
        make_result(
            changes=[
                ProposedCodeChange(
                    relative_path="src/views/UsersView.vue",
                    objective="wire page-size control",
                    proposed_approach="bind existing store state",
                )
            ]
        ),
        surface=make_surface(),
    )

    assert decision.action == "continue"


def test_runtime_policy_skips_confirmed_noop():
    decision = RuntimeExecutionDecisionPolicy().decide(
        make_result(),
        surface=make_surface(),
    )

    assert decision.action == "skip_runtime_noop"


def test_runtime_policy_blocks_when_critical_dependency_is_missing():
    decision = RuntimeExecutionDecisionPolicy().decide(
        make_result(
            missing=["src/api/users.ts 的内容未提供，" "无法确认请求协议是否兼容"]
        ),
        surface=make_surface(),
    )

    assert decision.action == "block_insufficient_evidence"


def test_runtime_policy_allows_reference_and_test_missing_information():
    decision = RuntimeExecutionDecisionPolicy().decide(
        make_result(
            gaps=[
                EvidenceGap(
                    description=(
                        "src/components/PaginationBar.vue "
                        "的完整内容未在 code_context 中提供"
                    ),
                    severity="advisory",
                    related_paths=[
                        "src/components/PaginationBar.vue",
                    ],
                    reason="参考组件信息缺失，但不阻塞当前修改判断",
                ),
                EvidenceGap(
                    description=(
                        "tests/unit/usersStore.test.ts "
                        "的内容未在 code_context 中提供"
                    ),
                    severity="advisory",
                    related_paths=[
                        "tests/unit/usersStore.test.ts",
                    ],
                    reason="测试覆盖信息缺失，但后续 Verification 会负责验证",
                ),
                EvidenceGap(
                    description=(
                        "repository.scripts 为空，" "无法确认 lint 和 build 命令"
                    ),
                    severity="advisory",
                    related_paths=[],
                    reason="Benchmark Verification Policy 已提供验证命令",
                ),
                EvidenceGap(
                    description=("尚未执行任何测试或构建，" "无法提供实际通过结果"),
                    severity="advisory",
                    related_paths=[],
                    reason="实际测试由后续 Verification Gate 执行",
                ),
            ]
        ),
        surface=make_surface(),
    )

    assert decision.action == "skip_runtime_noop"


def test_runtime_policy_still_blocks_unknown_missing_information():
    decision = RuntimeExecutionDecisionPolicy().decide(
        make_result(missing=["缺少关键业务约束，无法判断当前实现是否正确"]),
        surface=make_surface(),
    )

    assert decision.action == "block_insufficient_evidence"
