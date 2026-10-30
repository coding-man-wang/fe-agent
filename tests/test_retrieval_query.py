from agent_core.codebase.retrieval_query import (
    RetrievalQueryBuilder,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
)


def make_requirement_spec():
    return RequirementSpec(
        title=("优化用户搜索框提示"),
        summary=("让用户知道可以按姓名" "或邮箱搜索用户"),
        goals=["优化搜索提示"],
        non_goals=["不修改 API"],
        acceptance_criteria=["搜索框显示新的" " placeholder"],
        constraints=[
            "不修改 API",
            "不修改 Router",
        ],
        assumptions=[],
        open_questions=[],
    )


def make_task():
    return TaskDefinition(
        task_id="TASK-001",
        task_type="code_change",
        title="优化用户搜索提示",
        objective=("修改用户搜索输入框" "的提示内容"),
        source_step_order=1,
        depends_on=[],
        inputs=[],
        target_modules=["用户筛选组件"],
        expected_outputs=["新的搜索提示"],
        verification=["运行测试"],
        acceptance_criteria_refs=[],
    )


def test_raw_literal_is_preserved():

    builder = RetrievalQueryBuilder(max_queries=6)

    queries = builder.build(
        task=make_task(),
        raw_requirement=("当前 placeholder 为 " '"Search users"，' "请调整提示内容。"),
        requirement_spec=(make_requirement_spec()),
    )

    texts = [query.text for query in queries]

    assert "Search users" in texts

    query = next(item for item in queries if item.text == "Search users")

    assert "raw_requirement" in query.sources

    assert "literal" in query.kinds


def test_queries_include_multiple_sources():

    builder = RetrievalQueryBuilder(max_queries=6)

    queries = builder.build(
        task=make_task(),
        raw_requirement=('把 "Search users" ' "修改为新的搜索提示。"),
        requirement_spec=(make_requirement_spec()),
    )

    sources = {source for query in queries for source in query.sources}

    assert "raw_requirement" in sources

    assert "requirement_spec" in sources

    assert "task" in sources


def test_constraints_are_not_used_as_positive_queries():

    builder = RetrievalQueryBuilder(max_queries=10)

    queries = builder.build(
        task=make_task(),
        raw_requirement=("优化用户搜索提示。"),
        requirement_spec=(make_requirement_spec()),
    )

    texts = {query.text for query in queries}

    # RequirementSpec.constraints 中有：
    #
    # 不修改 API
    # 不修改 Router
    #
    # 它们属于限制条件，
    # 不应该被当作正向搜索词。

    assert "不修改 API" not in texts

    assert "不修改 Router" not in texts


def test_duplicate_query_merges_sources():

    requirement = RequirementSpec(
        title="用户搜索",
        summary="用户搜索",
        goals=[],
        non_goals=[],
        acceptance_criteria=[],
        constraints=[],
        assumptions=[],
        open_questions=[],
    )

    task = TaskDefinition(
        task_id="TASK-001",
        task_type="code_change",
        title="用户搜索",
        objective="用户搜索",
        source_step_order=1,
        depends_on=[],
        inputs=[],
        target_modules=[],
        expected_outputs=[],
        verification=[],
        acceptance_criteria_refs=[],
    )

    builder = RetrievalQueryBuilder(max_queries=6)

    queries = builder.build(
        task=task,
        raw_requirement=("处理用户搜索功能"),
        requirement_spec=(requirement),
    )

    query = next(item for item in queries if item.text == "用户搜索")

    assert set(query.sources) == {
        "requirement_spec",
        "task",
    }


def test_quoted_literal_can_generate_search_token():

    builder = RetrievalQueryBuilder(max_queries=10)

    queries = builder.build(
        task=make_task(),
        raw_requirement=('把 "Page 2 / 5" ' "改成新的分页格式。"),
        requirement_spec=(make_requirement_spec()),
    )

    texts = [query.text for query in queries]

    assert "Page 2 / 5" in texts

    # Exact Search 搜完整
    # Page 2 / 5 不一定命中：
    #
    # Page {{ page }} / {{ totalPages }}
    #
    # 所以额外保留 Page。
    assert "Page" in texts


def test_logical_task_queries_use_responsibilities_and_questions():
    builder = RetrievalQueryBuilder(max_queries=10)

    task = TaskDefinition(
        task_id="TASK-001",
        task_type="investigation",
        title="调查订单筛选",
        objective="确认 Runtime Path",
        source_step_order=1,
        responsibilities=[
            "确认筛选状态责任",
        ],
        investigation_questions=[
            "订单请求在哪里传播 status？",
        ],
        repository_hints=[
            "src/api",
        ],
    )

    queries = builder.build(
        task=task,
    )

    texts = {item.text for item in queries}

    assert "确认筛选状态责任" in texts

    assert "订单请求在哪里传播 status？" in texts

    assert "src/api" in texts
