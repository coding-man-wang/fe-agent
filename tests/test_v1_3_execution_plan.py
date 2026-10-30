from agent_core.benchmark.execution_model import (
    FullExecutionRunRecord,
    FullExecutionTaskRecord,
)
from agent_core.benchmark.execution_plan import (
    ExecutionPlanCompressor,
)
from agent_core.change_surface.model import (
    ResolvedChangeSurface,
)
from agent_core.sdd.model import (
    TaskDefinition,
    TaskPlan,
)
from agent_core.sdd.task_graph import (
    topological_sort_tasks,
)


def make_task(
    *,
    task_id: str,
    task_type: str,
    depends_on=None,
    criteria=None,
    responsibility=None,
):
    depends_on = depends_on or []

    criteria = criteria or []

    kwargs = {
        "task_id": task_id,
        "task_type": task_type,
        "title": (f"title {task_id}"),
        "objective": (f"objective {task_id}"),
        "source_step_order": 1,
        "depends_on": depends_on,
        "inputs": [],
        "target_modules": [],
        "expected_outputs": [f"output {task_id}"],
        "verification": [f"verify {task_id}"],
        "acceptance_criteria_refs": (criteria),
    }

    # v1.3 Logical TaskDefinition
    # 存在这些字段时补充。
    #
    # 这样测试同时兼容：
    # - legacy model
    # - v1.3 model
    fields = TaskDefinition.model_fields

    if "responsibilities" in fields:
        kwargs["responsibilities"] = [
            responsibility or (f"responsibility " f"{task_id}")
        ]

    if "investigation_questions" in fields:
        kwargs["investigation_questions"] = []

    if "repository_hints" in fields:
        kwargs["repository_hints"] = []

    return TaskDefinition(**kwargs)


def make_plan(
    tasks,
):
    kwargs = {
        "summary": ("execution compression test"),
        "tasks": tasks,
        "completion_criteria": ["all done"],
    }

    if "planning_stage" in TaskPlan.model_fields:
        kwargs["planning_stage"] = "logical"

    return TaskPlan(**kwargs)


def surface(
    task_id: str,
    targets,
    *,
    disposition=("change_required"),
):
    return ResolvedChangeSurface(
        task_id=task_id,
        implementation_targets=(list(targets)),
        dependencies=[],
        references=[],
        tests=[],
        create_files=[],
        unresolved=[],
        rationale=["test fixture"],
        execution_disposition=(disposition),
    )


def test_case005_like_code_tasks_are_compressed():

    tasks = [
        make_task(
            task_id="TASK-001",
            task_type="investigation",
        ),
        make_task(
            task_id="TASK-002",
            task_type="investigation",
        ),
        make_task(
            task_id="TASK-003",
            task_type="investigation",
            depends_on=["TASK-001"],
        ),
        make_task(
            task_id="TASK-004",
            task_type="code_change",
            depends_on=[
                "TASK-001",
                "TASK-002",
            ],
            criteria=[
                "AC-filter",
                "AC-mode",
            ],
            responsibility=("建立订单筛选完整生产闭环"),
        ),
        make_task(
            task_id="TASK-005",
            task_type="code_change",
            depends_on=[
                "TASK-001",
                "TASK-003",
                "TASK-004",
            ],
            criteria=[
                "AC-filter",
            ],
            responsibility=("页面接入筛选控件"),
        ),
        make_task(
            task_id="TASK-006",
            task_type="code_change",
            depends_on=[
                "TASK-001",
                "TASK-002",
                "TASK-004",
            ],
            criteria=[
                "AC-filter",
            ],
            responsibility=("数据获取流程传递筛选状态"),
        ),
        make_task(
            task_id="TASK-007",
            task_type="code_change",
            depends_on=[
                "TASK-002",
                "TASK-006",
            ],
            criteria=[
                "AC-mode",
            ],
            responsibility=("Remote 参数映射"),
        ),
        make_task(
            task_id="TASK-008",
            task_type="code_change",
            depends_on=[
                "TASK-002",
                "TASK-006",
            ],
            criteria=[
                "AC-filter",
            ],
            responsibility=("Mock 状态过滤"),
        ),
        make_task(
            task_id="TASK-009",
            task_type="code_change",
            depends_on=[
                "TASK-003",
                "TASK-006",
            ],
            criteria=[
                "AC-filter",
            ],
            responsibility=("处理快速切换竞态"),
        ),
        make_task(
            task_id="TASK-010",
            task_type="test",
            depends_on=[
                "TASK-004",
                "TASK-005",
                "TASK-006",
                "TASK-007",
                "TASK-008",
                "TASK-009",
            ],
            criteria=[
                "AC-filter",
            ],
        ),
        make_task(
            task_id="TASK-011",
            task_type="verification",
            depends_on=["TASK-010"],
        ),
    ]

    plan = make_plan(tasks)

    production_targets = [
        "src/views/OrdersView.vue",
        "src/api/orders.ts",
        "src/mocks/service.ts",
    ]

    surfaces = {
        "TASK-004": surface(
            "TASK-004",
            production_targets,
        ),
        "TASK-005": surface(
            "TASK-005",
            production_targets,
        ),
        "TASK-006": surface(
            "TASK-006",
            production_targets,
        ),
        "TASK-007": surface(
            "TASK-007",
            [
                "src/api/orders.ts",
            ],
        ),
        "TASK-008": surface(
            "TASK-008",
            [
                "src/mocks/service.ts",
            ],
        ),
        "TASK-009": surface(
            "TASK-009",
            [
                "src/views/OrdersView.vue",
            ],
        ),
        "TASK-010": (
            ResolvedChangeSurface(
                task_id="TASK-010",
                implementation_targets=[],
                dependencies=[
                    *production_targets,
                ],
                references=[],
                tests=["tests/unit/" "ordersFilter.test.ts"],
                create_files=["tests/unit/" "ordersFilter.test.ts"],
                unresolved=[],
                rationale=["test fixture"],
                execution_disposition=("change_required"),
            )
        ),
    }

    result = ExecutionPlanCompressor().compress(
        task_plan=plan,
        task_surfaces=surfaces,
    )

    assert result.summary.logical_task_count == 11

    assert result.summary.execution_task_count == 6

    assert result.summary.coalesced_task_count == 5

    assert len(result.summary.groups) == 1

    group = result.summary.groups[0]

    assert group.leader_task_id == "TASK-004"

    assert group.member_task_ids == [
        "TASK-004",
        "TASK-005",
        "TASK-006",
        "TASK-007",
        "TASK-008",
        "TASK-009",
    ]

    assert result.coalesced_task_ids_for_leader("TASK-004") == [
        "TASK-005",
        "TASK-006",
        "TASK-007",
        "TASK-008",
        "TASK-009",
    ]

    execution_map = {task.task_id: task for task in (result.execution_task_plan.tasks)}

    # Followers 已经不再进入
    # Runtime Execution DAG。
    for task_id in [
        "TASK-005",
        "TASK-006",
        "TASK-007",
        "TASK-008",
        "TASK-009",
    ]:
        assert task_id not in execution_map

    leader = execution_map["TASK-004"]

    # TASK-003 原本只属于
    # downstream logical task。
    #
    # Compression 后必须补到
    # leader external dependency。
    assert set(leader.depends_on) == {
        "TASK-001",
        "TASK-002",
        "TASK-003",
    }

    # Test Task 原来依赖六个
    # production logical tasks，
    # 现在应该只依赖一个
    # Execution Unit。
    assert execution_map["TASK-010"].depends_on == ["TASK-004"]

    if "responsibilities" in TaskDefinition.model_fields:
        responsibilities = leader.responsibilities

        assert "页面接入筛选控件" in responsibilities

        assert "Remote 参数映射" in responsibilities

        assert "Mock 状态过滤" in responsibilities

        assert "处理快速切换竞态" in responsibilities

    # 最终 DAG 必须仍合法。
    ordered = topological_sort_tasks(result.execution_task_plan)

    assert [item.task_id for item in ordered] == [
        "TASK-001",
        "TASK-002",
        "TASK-003",
        "TASK-004",
        "TASK-010",
        "TASK-011",
    ]


def test_unrelated_tasks_with_same_files_are_not_merged():

    plan = make_plan(
        [
            make_task(
                task_id="TASK-001",
                task_type=("investigation"),
            ),
            make_task(
                task_id="TASK-002",
                task_type="code_change",
                depends_on=["TASK-001"],
                criteria=["AC-1"],
            ),
            make_task(
                task_id="TASK-003",
                task_type="code_change",
                depends_on=["TASK-001"],
                criteria=["AC-1"],
            ),
        ]
    )

    surfaces = {
        "TASK-002": surface(
            "TASK-002",
            ["src/a.ts"],
        ),
        "TASK-003": surface(
            "TASK-003",
            ["src/a.ts"],
        ),
    }

    result = ExecutionPlanCompressor().compress(
        task_plan=plan,
        task_surfaces=surfaces,
    )

    # 虽然修改同一个文件，
    # 但 TASK-003 不是 TASK-002
    # downstream。
    assert result.summary.coalesced_task_count == 0


def test_follower_cannot_expand_leader_write_surface():

    plan = make_plan(
        [
            make_task(
                task_id="TASK-001",
                task_type=("investigation"),
            ),
            make_task(
                task_id="TASK-002",
                task_type="code_change",
                depends_on=["TASK-001"],
                criteria=["AC-1"],
            ),
            make_task(
                task_id="TASK-003",
                task_type="code_change",
                depends_on=["TASK-002"],
                criteria=["AC-1"],
            ),
        ]
    )

    surfaces = {
        "TASK-002": surface(
            "TASK-002",
            ["src/a.ts"],
        ),
        "TASK-003": surface(
            "TASK-003",
            [
                "src/a.ts",
                "src/b.ts",
            ],
        ),
    }

    result = ExecutionPlanCompressor().compress(
        task_plan=plan,
        task_surfaces=surfaces,
    )

    assert result.summary.coalesced_task_count == 0


def test_insufficient_evidence_task_is_not_merged():

    plan = make_plan(
        [
            make_task(
                task_id="TASK-001",
                task_type=("investigation"),
            ),
            make_task(
                task_id="TASK-002",
                task_type="code_change",
                depends_on=["TASK-001"],
                criteria=["AC-1"],
            ),
            make_task(
                task_id="TASK-003",
                task_type="code_change",
                depends_on=["TASK-002"],
                criteria=["AC-1"],
            ),
        ]
    )

    surfaces = {
        "TASK-002": surface(
            "TASK-002",
            ["src/a.ts"],
        ),
        "TASK-003": surface(
            "TASK-003",
            ["src/a.ts"],
            disposition=("insufficient_evidence"),
        ),
    }

    result = ExecutionPlanCompressor().compress(
        task_plan=plan,
        task_surfaces=surfaces,
    )

    assert result.summary.coalesced_task_count == 0


def test_code_change_is_not_merged_across_test_boundary():

    plan = make_plan(
        [
            make_task(
                task_id="TASK-001",
                task_type=("investigation"),
            ),
            make_task(
                task_id="TASK-002",
                task_type="code_change",
                depends_on=["TASK-001"],
                criteria=["AC-1"],
            ),
            make_task(
                task_id="TASK-003",
                task_type="test",
                depends_on=["TASK-002"],
                criteria=["AC-1"],
            ),
            make_task(
                task_id="TASK-004",
                task_type="code_change",
                depends_on=["TASK-003"],
                criteria=["AC-1"],
            ),
        ]
    )

    surfaces = {
        "TASK-002": surface(
            "TASK-002",
            ["src/a.ts"],
        ),
        "TASK-003": (
            ResolvedChangeSurface(
                task_id="TASK-003",
                implementation_targets=[],
                dependencies=["src/a.ts"],
                references=[],
                tests=["tests/a.test.ts"],
                create_files=[],
                unresolved=[],
                rationale=[],
                execution_disposition=("change_required"),
            )
        ),
        "TASK-004": surface(
            "TASK-004",
            ["src/a.ts"],
        ),
    }

    result = ExecutionPlanCompressor().compress(
        task_plan=plan,
        task_surfaces=surfaces,
    )

    assert result.summary.coalesced_task_count == 0


def test_full_execution_models_accept_compression_observability():

    plan = make_plan([])

    compression = ExecutionPlanCompressor().compress(
        task_plan=plan,
        task_surfaces={},
    )

    record = FullExecutionRunRecord(
        run_id="run-1",
        baseline_run_id=("baseline-1"),
        case_id="CASE-X",
        base_commit_sha="abc",
        execution_plan_compression=(compression.summary),
    )

    assert record.execution_plan_compression is not None

    task_record = FullExecutionTaskRecord(
        task_id="TASK-004",
        task_type="code_change",
        coalesced_task_ids=[
            "TASK-005",
            "TASK-006",
        ],
        status="completed",
    )

    assert task_record.coalesced_task_ids == [
        "TASK-005",
        "TASK-006",
    ]


def test_same_surface_but_different_contract_is_not_merged():
    plan = make_plan(
        [
            make_task(
                task_id="TASK-001",
                task_type="investigation",
            ),
            make_task(
                task_id="TASK-002",
                task_type="code_change",
                depends_on=["TASK-001"],
                criteria=["AC-feature-a"],
            ),
            make_task(
                task_id="TASK-003",
                task_type="code_change",
                depends_on=["TASK-002"],
                criteria=["AC-feature-b"],
            ),
        ]
    )

    surfaces = {
        "TASK-002": surface(
            "TASK-002",
            ["src/a.ts"],
        ),
        "TASK-003": surface(
            "TASK-003",
            ["src/a.ts"],
        ),
    }

    result = ExecutionPlanCompressor().compress(
        task_plan=plan,
        task_surfaces=surfaces,
    )

    assert result.summary.coalesced_task_count == 0
