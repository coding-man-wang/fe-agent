from agent_core.benchmark.evidence_rerank_filter import (
    EvidenceProtectedRerankFilter,
)
from agent_core.benchmark.task_rerank_diagnostics import (
    RerankFileScore,
    TaskRerankDiagnosticResult,
)
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)

PATHS = [
    "src/api/orders.ts",
    "src/views/OrdersView.vue",
    "src/mocks/service.ts",
    "src/components/UserFilters.vue",
    "src/types/domain.ts",
    "src/api/requestClient.ts",
]


def make_selection():
    files = []

    for rank, path in enumerate(PATHS, start=1):
        is_user_filter = path.endswith("UserFilters.vue")
        is_request = path.endswith("requestClient.ts")

        symbols = []

        if not is_user_filter:
            symbols = [
                SymbolCandidate(
                    symbol_id=f"{path}::function::target",
                    name="target",
                    kind="function",
                    relative_path=path,
                    start_line=1,
                    end_line=1,
                    score=0.9,
                    source="evidence",
                )
            ]

        files.append(
            HybridRelevantFile(
                relative_path=path,
                category="source",
                language="TypeScript",
                size_bytes=100,
                retrieval_score=1.0,
                retrieval_rank=rank,
                policy_score=0,
                selection_score=1.0,
                semantic_hit_count=(4 if is_user_filter else 1),
                context_role=("dependency" if is_request else "unclassified"),
                symbol_candidates=symbols,
            )
        )

    return HybridRelevantFileSelection(
        candidate_count=6,
        eligible_count=6,
        selected_count=6,
        total_selected_bytes=600,
        selected_files=files,
    )


def make_diagnostic(selection, *, task="order"):
    if task == "order":
        ranking = [
            ("src/mocks/service.ts", 0.8043),
            ("src/views/OrdersView.vue", 0.7541),
            ("src/types/domain.ts", 0.7540),
            ("src/api/orders.ts", 0.7340),
            ("src/components/UserFilters.vue", 0.6330),
            ("src/api/requestClient.ts", 0.3369),
        ]
    else:
        ranking = [
            ("src/mocks/service.ts", 0.8137),
            ("src/components/UserFilters.vue", 0.8074),
            ("src/types/domain.ts", 0.7774),
            ("src/api/orders.ts", 0.5123),
            ("src/views/OrdersView.vue", 0.5044),
            ("src/api/requestClient.ts", 0.3012),
        ]

    by_path = {item.relative_path: item for item in selection.selected_files}

    return TaskRerankDiagnosticResult(
        query=task,
        status="complete",
        candidate_paths=PATHS,
        ranked_files=[
            RerankFileScore(
                relative_path=path,
                rank=rank,
                score=score,
                context_role=by_path[path].context_role,
                has_symbol_evidence=bool(by_path[path].symbol_candidates),
                source_truncated=False,
            )
            for rank, (path, score) in enumerate(ranking, start=1)
        ],
    )


def test_order_task_proposes_user_filters_only():
    selection = make_selection()
    original = selection.model_dump(mode="json")

    proposal = EvidenceProtectedRerankFilter().propose(
        selection=selection,
        diagnostic=make_diagnostic(selection),
    )

    assert proposal.status == "proposed"
    assert proposal.proposed_drops == ["src/components/UserFilters.vue"]

    paths = [item.relative_path for item in proposal.selection.selected_files]

    assert "src/api/requestClient.ts" in paths
    assert proposal.selection.selected_count == 5

    # Shadow 不修改输入
    assert selection.model_dump(mode="json") == original


def test_user_task_keeps_user_filters():
    selection = make_selection()

    proposal = EvidenceProtectedRerankFilter().propose(
        selection=selection,
        diagnostic=make_diagnostic(
            selection,
            task="user",
        ),
    )

    assert proposal.status == "no_change"
    assert proposal.proposed_drops == []
    assert proposal.selection.selected_count == 6


def test_incomplete_diagnostic_fails_safe():
    selection = make_selection()
    report = make_diagnostic(selection)

    report.status = "incomplete"

    proposal = EvidenceProtectedRerankFilter().propose(
        selection=selection,
        diagnostic=report,
    )

    assert proposal.status == "abstained"
    assert proposal.proposed_drops == []
    assert proposal.selection.selected_count == 6


def test_selection_mismatch_fails_safe():
    selection = make_selection()
    report = make_diagnostic(selection)

    report.candidate_paths = PATHS[:-1]

    proposal = EvidenceProtectedRerankFilter().propose(
        selection=selection,
        diagnostic=report,
    )

    assert proposal.status == "abstained"
    assert proposal.proposed_drops == []


def test_truncated_source_is_protected():
    selection = make_selection()
    report = make_diagnostic(selection)

    for item in report.ranked_files:
        if item.relative_path.endswith("UserFilters.vue"):
            item.source_truncated = True

    proposal = EvidenceProtectedRerankFilter().propose(
        selection=selection,
        diagnostic=report,
    )

    assert proposal.proposed_drops == []
