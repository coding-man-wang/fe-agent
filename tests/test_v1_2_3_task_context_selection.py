import asyncio
from unittest.mock import (
    AsyncMock,
    Mock,
)

from agent_core.codebase.task_context_selection import (
    TaskContextSelectionService,
)


def test_select_rebuilds_current_task_context(
    tmp_path,
):
    index = Mock()
    retrieval_selection = Mock()
    graph = Mock()
    final_selection = Mock()

    file_indexer = Mock()
    file_indexer.build.return_value = index

    retrieval_service = Mock()

    query_specs = [Mock()]

    retrieval_service.build_query_specs.return_value = query_specs

    retrieval_service.retrieve = AsyncMock(return_value=(retrieval_selection))

    graph_builder = Mock()
    graph_builder.build.return_value = graph

    graph_expander = Mock()
    graph_expander.expand.return_value = final_selection

    service = TaskContextSelectionService(
        file_indexer=file_indexer,
        retrieval_service=(retrieval_service),
        graph_builder=graph_builder,
        graph_expander=graph_expander,
    )

    task = Mock()
    requirement_spec = Mock()

    result = asyncio.run(
        service.select(
            repository_root=tmp_path,
            repository_revision="commit-1",
            task=task,
            raw_requirement=("支持分页大小切换"),
            requirement_spec=(requirement_spec),
        )
    )

    file_indexer.build.assert_called_once_with(tmp_path.resolve())

    (
        retrieval_service.build_query_specs.assert_called_once_with(
            task=task,
            raw_requirement=("支持分页大小切换"),
            requirement_spec=(requirement_spec),
        )
    )

    retrieval_service.retrieve.assert_awaited_once_with(
        task=task,
        index=index,
        query_specs=query_specs,
        semantic_enabled=False,
    )

    graph_builder.build.assert_called_once_with(
        file_index=index,
        repository_revision="commit-1",
    )

    graph_expander.expand.assert_called_once_with(
        selection=retrieval_selection,
        index=index,
        graph=graph,
    )

    assert result.repository_revision == "commit-1"

    assert result.index is index

    assert result.retrieval_selection is retrieval_selection

    assert result.graph is graph

    assert result.selection is final_selection


def test_semantic_index_rebuilds_only_when_revision_changes(
    tmp_path,
):
    index = Mock()

    file_indexer = Mock()
    file_indexer.build.return_value = index

    retrieval_selection = Mock()
    final_selection = Mock()
    graph = Mock()

    retrieval_service = Mock()

    retrieval_service.build_query_specs.return_value = [Mock()]

    retrieval_service.retrieve = AsyncMock(return_value=(retrieval_selection))

    graph_builder = Mock()
    graph_builder.build.return_value = graph

    graph_expander = Mock()
    graph_expander.expand.return_value = final_selection

    chunks = [
        Mock(),
        Mock(),
    ]

    code_chunker = Mock()

    code_chunker.build_chunks.return_value = chunks

    semantic_index = Mock()

    semantic_index.build = AsyncMock()

    service = TaskContextSelectionService(
        file_indexer=file_indexer,
        retrieval_service=(retrieval_service),
        graph_builder=graph_builder,
        graph_expander=graph_expander,
        code_chunker=code_chunker,
        semantic_index=semantic_index,
    )

    task = Mock()
    requirement_spec = Mock()

    async def run():
        # 第一次 commit-1：
        # 应该 Build Semantic Index。
        await service.select(
            repository_root=tmp_path,
            repository_revision="commit-1",
            task=task,
            raw_requirement="requirement",
            requirement_spec=(requirement_spec),
        )

        # Commit 没变：
        # 不应该重新 Build。
        await service.select(
            repository_root=tmp_path,
            repository_revision="commit-1",
            task=task,
            raw_requirement="requirement",
            requirement_spec=(requirement_spec),
        )

        # Commit 变化：
        # 必须重新 Build。
        await service.select(
            repository_root=tmp_path,
            repository_revision="commit-2",
            task=task,
            raw_requirement="requirement",
            requirement_spec=(requirement_spec),
        )

    asyncio.run(run())

    assert semantic_index.build.await_count == 2

    assert code_chunker.build_chunks.call_count == 2


def test_semantic_components_must_be_configured_together():
    file_indexer = Mock()
    retrieval_service = Mock()

    try:
        TaskContextSelectionService(
            file_indexer=file_indexer,
            retrieval_service=(retrieval_service),
            code_chunker=Mock(),
            semantic_index=None,
        )

    except ValueError:
        pass

    else:
        raise AssertionError("只配置 code_chunker " "不应允许创建 Service")
