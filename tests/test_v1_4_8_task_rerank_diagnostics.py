from pathlib import Path

import pytest

from agent_core.benchmark.task_rerank_diagnostics import (
    TaskRerankDiagnosticRunner,
)
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
)
from agent_core.codebase.reranker import (
    Reranker,
    RerankHit,
)


class FakeReranker(Reranker):
    def __init__(
        self,
        scores: dict[str, float],
        *,
        incomplete: bool = False,
        fail: bool = False,
    ):
        self.scores = scores
        self.incomplete = incomplete
        self.fail = fail
        self.calls = []

    async def rerank(
        self,
        *,
        query,
        documents,
        top_n,
    ):
        self.calls.append(
            {
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }
        )

        if self.fail:
            raise RuntimeError("simulated provider failure")

        ranked = sorted(
            documents,
            key=lambda doc: (
                -self.scores[doc.document_id],
                doc.document_id,
            ),
        )

        if self.incomplete:
            ranked = ranked[:-1]

        return [
            RerankHit(
                document_id=doc.document_id,
                rank=rank,
                score=self.scores[doc.document_id],
            )
            for rank, doc in enumerate(ranked, start=1)
        ]


def make_selection(root: Path):
    files = []

    contents = {
        "src/api/orders.ts": ("export function listOrders() { return [] }\n"),
        "src/components/UserFilters.vue": (
            "<template><select>User status</select></template>\n"
        ),
    }

    for rank, (relative_path, content) in enumerate(
        contents.items(),
        start=1,
    ):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        files.append(
            HybridRelevantFile(
                relative_path=relative_path,
                category="source",
                language="TypeScript",
                size_bytes=len(content.encode("utf-8")),
                retrieval_score=1.0,
                retrieval_rank=rank,
                policy_score=0,
                selection_score=1.0,
                semantic_hit_count=1,
                context_role="unclassified",
            )
        )

    return HybridRelevantFileSelection(
        candidate_count=len(files),
        eligible_count=len(files),
        selected_count=len(files),
        total_selected_bytes=sum(file.size_bytes for file in files),
        selected_files=files,
    )


@pytest.mark.asyncio
async def test_complete_rerank_keeps_original_selection(
    tmp_path,
):
    selection = make_selection(tmp_path)
    original = selection.model_dump(mode="json")

    reranker = FakeReranker(
        {
            "src/api/orders.ts": 0.95,
            "src/components/UserFilters.vue": 0.12,
        }
    )

    report = await TaskRerankDiagnosticRunner(
        reranker=reranker,
    ).run(
        query="为订单列表增加状态筛选",
        repository_root=tmp_path,
        selection=selection,
    )

    assert report.status == "complete"
    assert report.issues == []

    assert [item.relative_path for item in report.ranked_files] == [
        "src/api/orders.ts",
        "src/components/UserFilters.vue",
    ]

    assert reranker.calls[0]["top_n"] == 2

    # 诊断不得改变正式 Selection
    assert selection.model_dump(mode="json") == original


@pytest.mark.asyncio
async def test_incomplete_provider_response_is_rejected(
    tmp_path,
):
    selection = make_selection(tmp_path)

    reranker = FakeReranker(
        {
            "src/api/orders.ts": 0.95,
            "src/components/UserFilters.vue": 0.12,
        },
        incomplete=True,
    )

    report = await TaskRerankDiagnosticRunner(
        reranker=reranker,
    ).run(
        query="订单筛选",
        repository_root=tmp_path,
        selection=selection,
    )

    assert report.status == "incomplete"
    assert "incomplete_rerank_response" in report.issues
    assert report.ranked_files == []


@pytest.mark.asyncio
async def test_provider_failure_does_not_change_selection(
    tmp_path,
):
    selection = make_selection(tmp_path)
    original = selection.model_dump(mode="json")

    reranker = FakeReranker({}, fail=True)

    report = await TaskRerankDiagnosticRunner(
        reranker=reranker,
    ).run(
        query="订单筛选",
        repository_root=tmp_path,
        selection=selection,
    )

    assert report.status == "error"
    assert report.ranked_files == []
    assert selection.model_dump(mode="json") == original


@pytest.mark.asyncio
async def test_same_file_can_receive_different_task_ranks(
    tmp_path,
):
    selection = make_selection(tmp_path)

    order_reranker = FakeReranker(
        {
            "src/api/orders.ts": 0.95,
            "src/components/UserFilters.vue": 0.12,
        }
    )

    user_reranker = FakeReranker(
        {
            "src/api/orders.ts": 0.15,
            "src/components/UserFilters.vue": 0.96,
        }
    )

    order = await TaskRerankDiagnosticRunner(
        reranker=order_reranker,
    ).run(
        query="订单列表状态筛选",
        repository_root=tmp_path,
        selection=selection,
    )

    user = await TaskRerankDiagnosticRunner(
        reranker=user_reranker,
    ).run(
        query="用户列表状态筛选",
        repository_root=tmp_path,
        selection=selection,
    )

    assert order.status == "complete"
    assert user.status == "complete"

    assert order.ranked_files[0].relative_path == ("src/api/orders.ts")

    assert user.ranked_files[0].relative_path == ("src/components/UserFilters.vue")
