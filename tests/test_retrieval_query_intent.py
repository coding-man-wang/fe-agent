from agent_core.codebase.retrieval_query import (
    RetrievalQueryBuilder,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


def make_task(
    *,
    task_type: str,
    title: str,
    objective: str,
    inputs: list[str] | None = None,
    target_modules: list[str] | None = None,
) -> TaskDefinition:

    return TaskDefinition(
        task_id="TASK-001",
        task_type=task_type,
        title=title,
        objective=objective,
        source_step_order=1,
        depends_on=[],
        inputs=inputs or [],
        target_modules=target_modules or [],
        expected_outputs=[],
        verification=[],
        acceptance_criteria_refs=[],
    )


def test_code_change_defaults_to_implementation_location():
    builder = RetrievalQueryBuilder(
        max_queries=10,
    )

    task = make_task(
        task_type="code_change",
        title="修改用户分页",
        objective="增加 pageSize 切换能力",
    )

    queries = builder.build(
        task=task,
    )

    title_query = next(query for query in queries if query.text == "修改用户分页")

    assert "implementation_location" in title_query.intents


def test_reuse_text_adds_similar_implementation():
    builder = RetrievalQueryBuilder(
        max_queries=10,
    )

    task = make_task(
        task_type="code_change",
        title="修改用户分页",
        objective=("复用现有 pageSize 实现"),
    )

    queries = builder.build(
        task=task,
    )

    objective_query = next(
        query for query in queries if query.text == "复用现有 pageSize 实现"
    )

    assert "implementation_location" in objective_query.intents

    assert "similar_implementation" in objective_query.intents


def test_test_task_uses_test_pattern():
    builder = RetrievalQueryBuilder(
        max_queries=10,
    )

    task = make_task(
        task_type="test",
        title="增加分页 E2E 测试",
        objective=("使用 Playwright 验证 pageSize"),
    )

    queries = builder.build(
        task=task,
    )

    for query in queries:

        assert "test_pattern" in query.intents


def test_same_query_merges_multiple_intents():
    builder = RetrievalQueryBuilder(
        max_queries=10,
    )

    text = "检查用户分页实现"

    task = make_task(
        task_type="investigation",
        title=text,
        objective="理解分页调用链",
    )

    queries = builder.build(
        task=task,
        raw_requirement=text,
    )

    merged = next(query for query in queries if query.text == text)

    assert "implementation_location" in merged.intents

    assert "dependency_context" in merged.intents
