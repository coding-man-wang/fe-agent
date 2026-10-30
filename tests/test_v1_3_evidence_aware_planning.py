from agent_core.artifacts.resolver import (
    TaskArtifactResolver,
)
from agent_core.artifacts.store import (
    JsonTaskArtifactStore,
)
from agent_core.change_surface.model import (
    ResolvedChangeSurface,
)
from agent_core.change_surface.replan import (
    RepositoryAwareReplanner,
)
from agent_core.sdd.model import (
    ImplementationPlan,
    ImplementationStep,
    ProjectContext,
    RequirementSpec,
    TaskDefinition,
    TaskPlan,
)
from agent_core.sdd.task_validation import (
    validate_task_plan,
)


def _requirement():
    return RequirementSpec(
        title="订单筛选",
        summary="支持订单状态筛选",
        acceptance_criteria=[
            "切换状态后重新加载订单",
        ],
    )


def _implementation_plan():
    return ImplementationPlan(
        summary="实现订单筛选",
        steps=[
            ImplementationStep(
                order=1,
                title="确认现有实现",
                goal="确认当前 Runtime Path",
            ),
            ImplementationStep(
                order=2,
                title="实现行为",
                goal="完成筛选功能",
            ),
            ImplementationStep(
                order=3,
                title="最终验证",
                goal="验证需求",
            ),
        ],
    )


def _project_context():
    return ProjectContext(
        project_name="demo",
        framework="Vue3",
        language="TypeScript",
        directory_summary=[
            "src/api - API 访问层",
            "src/views - 页面层",
            "tests/unit - 单元测试",
        ],
    )


def _logical_plan():
    return TaskPlan(
        planning_stage="logical",
        summary="Logical Plan",
        tasks=[
            TaskDefinition(
                task_id="TASK-001",
                task_type="investigation",
                title="确认筛选 Runtime Path",
                objective="确认现有筛选责任分布",
                source_step_order=1,
                responsibilities=[
                    "确认筛选状态和请求传播责任",
                ],
                investigation_questions=[
                    "当前筛选状态由哪一层维护？",
                    "请求参数由哪一层传播？",
                ],
                repository_hints=[
                    "src/views",
                    "src/api",
                ],
                expected_outputs=[
                    "Repository Evidence",
                ],
                verification=[
                    "调查结果必须引用真实代码事实",
                ],
                acceptance_criteria_refs=[
                    "切换状态后重新加载订单",
                ],
            ),
            TaskDefinition(
                task_id="TASK-002",
                task_type="code_change",
                title="实现订单筛选",
                objective="实现筛选和重新加载行为",
                source_step_order=2,
                depends_on=[
                    "TASK-001",
                ],
                responsibilities=[
                    "实现筛选状态传播和 reload",
                ],
                repository_hints=[
                    "src/views",
                    "src/api",
                ],
                expected_outputs=[
                    "满足需求的生产代码行为",
                ],
                verification=[
                    "代码通过相关验证",
                ],
                acceptance_criteria_refs=[
                    "切换状态后重新加载订单",
                ],
            ),
            TaskDefinition(
                task_id="TASK-003",
                task_type="verification",
                title="最终验证",
                objective="验证需求完成",
                source_step_order=3,
                depends_on=[
                    "TASK-002",
                ],
                expected_outputs=[
                    "Verification Evidence",
                ],
                verification=[
                    "最终质量门禁全部通过",
                ],
                acceptance_criteria_refs=[
                    "切换状态后重新加载订单",
                ],
            ),
        ],
    )


def test_project_context_exposes_only_safe_repository_hints():
    context = _project_context()

    assert context.repository_hint_candidates() == [
        "src/api",
        "src/views",
        "tests/unit",
    ]


def test_logical_plan_contract_passes():
    result = validate_task_plan(
        _logical_plan(),
        _requirement(),
        _implementation_plan(),
        _project_context(),
    )

    assert result.is_valid is True

    assert result.errors == []


def test_logical_plan_rejects_legacy_target_modules():
    plan = _logical_plan()

    task = plan.tasks[1]

    plan = plan.model_copy(deep=True)

    plan.tasks[1] = task.model_copy(update={"target_modules": ["src/views"]})

    result = validate_task_plan(
        plan,
        _requirement(),
        _implementation_plan(),
        _project_context(),
    )

    assert result.is_valid is False

    assert any("target_modules" in item for item in result.errors)


def test_logical_plan_rejects_fabricated_repository_hint():
    plan = _logical_plan()

    task = plan.tasks[0]

    plan = plan.model_copy(deep=True)

    plan.tasks[0] = task.model_copy(update={"repository_hints": ["src/not-real"]})

    result = validate_task_plan(
        plan,
        _requirement(),
        _implementation_plan(),
        _project_context(),
    )

    assert result.is_valid is False

    assert any("src/not-real" in item for item in result.errors)


def test_code_change_requires_investigation_ancestor():
    plan = _logical_plan()

    code_task = plan.tasks[1]

    plan = plan.model_copy(deep=True)

    plan.tasks[1] = code_task.model_copy(update={"depends_on": []})

    result = validate_task_plan(
        plan,
        _requirement(),
        _implementation_plan(),
        _project_context(),
    )

    assert result.is_valid is False

    assert any("investigation ancestor" in item for item in result.errors)


def test_repository_search_hints_keep_legacy_compatibility():
    task = TaskDefinition(
        task_id="TASK-001",
        task_type="code_change",
        title="legacy",
        objective="legacy",
        source_step_order=1,
        repository_hints=[
            "src/views",
        ],
        target_modules=[
            "src/api",
            "src/views",
        ],
    )

    assert task.repository_search_hints() == [
        "src/views",
        "src/api",
    ]


def test_repository_aware_replan_preserves_logical_intent(
    tmp_path,
):
    plan = _logical_plan()

    store = JsonTaskArtifactStore(root=tmp_path / "artifacts")

    store.put_model(
        task_id="TASK-002",
        artifact_type=("change_surface"),
        model=(
            ResolvedChangeSurface(
                task_id="TASK-002",
                implementation_targets=[
                    "src/views/OrdersView.vue",
                    "src/api/orders.ts",
                ],
                dependencies=[],
                references=[],
                tests=[],
                unresolved=[],
                rationale=["Repository Evidence " "确认两个 Runtime Path " "需要修改"],
                execution_disposition=("change_required"),
            )
        ),
        depends_on_task_ids=[
            "TASK-001",
        ],
    )

    replanner = RepositoryAwareReplanner(
        artifact_resolver=(TaskArtifactResolver(store=store))
    )

    result = replanner.build(
        task_plan=plan,
        repository_revision=("abc123"),
    )

    assert result.source_planning_stage == "logical"

    assert result.repository_revision == "abc123"

    investigation = result.tasks[0]

    code_change = result.tasks[1]

    verification = result.tasks[2]

    assert investigation.resolution_status == "resolved"

    assert verification.resolution_status == "resolved"

    assert code_change.resolved_target_files == [
        "src/views/OrdersView.vue",
        "src/api/orders.ts",
    ]

    assert code_change.responsibilities == ["实现筛选状态传播和 reload"]

    assert code_change.repository_revision == "abc123"
