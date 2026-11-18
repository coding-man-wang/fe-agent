from copy import deepcopy
from types import SimpleNamespace

from agent_core.benchmark.d5_candidate_evidence import (
    analyze_d5_candidates,
)
from agent_core.benchmark.d5_effectiveness import (
    evaluate_d5_effectiveness,
)
from agent_core.change_surface.model import (
    ResolvedChangeSurface,
)
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
)
from agent_core.codebase.graph.model import (
    GraphFileNode,
    RepositoryDependencyEdge,
    RepositoryDependencyGraph,
)


def file(path, rank, *, graph=False, semantic=False, category="source"):
    return HybridRelevantFile(
        relative_path=path,
        category=category,
        size_bytes=100,
        retrieval_score=1.0,
        retrieval_rank=rank,
        policy_score=0,
        selection_score=1.0,
        context_role="dependency" if graph else "unclassified",
        lexical_hit_count=0 if graph or semantic else 1,
        semantic_hit_count=1 if semantic else 0,
        query_hit_count=1 if semantic else 0,
        reasons=(
            ["Graph dependency: external"]
            if graph
            else ["Semantic Retrieval"] if semantic else ["Lexical Retrieval"]
        ),
    )


def selection(files):
    return HybridRelevantFileSelection(
        candidate_count=len(files),
        eligible_count=len(files),
        selected_count=len(files),
        total_selected_bytes=sum(f.size_bytes for f in files),
        selected_files=files,
    )


def inspect(files, surface, *, graph=None, retrieval_count=5):
    if graph is None:
        graph = RepositoryDependencyGraph(
            root_path=".",
            nodes=[
                GraphFileNode(
                    relative_path=f.relative_path,
                    category=f.category,
                )
                for f in files
            ],
        )

    result = SimpleNamespace(
        repository_revision="a" * 40,
        selection=selection(files),
        retrieval_selection=selection(files[:retrieval_count]),
        graph=graph,
    )

    bundle = SimpleNamespace(
        task=SimpleNamespace(task_id="TASK-002"),
        code_context=[
            SimpleNamespace(
                relative_path=f.relative_path,
                category=f.category,
            )
            for f in files
        ],
        verification_strategy=None,
    )

    before = deepcopy(result.selection.model_dump())

    report = analyze_d5_candidates(
        selection_result=result,
        final_bundle=bundle,
        surface=surface,
    )

    assert result.selection.model_dump() == before
    assert report["applied"] is False
    assert report["active_execution_allowed"] is False

    return report


def test_case006_no_source_drop_candidate():
    files = [
        file("src/stores/users.ts", 1),
        file("src/views/UsersView.vue", 2),
        file("src/api/users.ts", 3),
        file("src/mocks/service.ts", 4),
        file("tests/unit/usersStore.test.ts", 5, category="test"),
        file("tests/e2e/users.spec.ts", 6, category="test"),
    ]

    surface = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=["src/stores/users.ts"],
        dependencies=[
            "src/views/UsersView.vue",
            "src/api/users.ts",
            "src/mocks/service.ts",
        ],
        tests=["tests/unit/usersStore.test.ts"],
    )

    report = inspect(files, surface)

    assert report["status"] == "no_source_drop_candidate"
    assert report["possible_candidate_paths"] == []


def test_unprotected_graph_only_candidate_is_discovered():
    files = [file(f"src/anchor{i}.ts", i + 1) for i in range(5)]
    files.append(file("src/weak.ts", 6, graph=True))

    graph = RepositoryDependencyGraph(
        root_path=".",
        nodes=[
            *[
                GraphFileNode(
                    relative_path=f.relative_path,
                    category="source",
                )
                for f in files
            ],
            GraphFileNode(
                relative_path="src/external.ts",
                category="source",
            ),
        ],
        edges=[
            RepositoryDependencyEdge(
                source_path="src/external.ts",
                target_path="src/weak.ts",
                import_kind="value",
                specifier="./weak",
                line_number=1,
            ),
        ],
    )

    surface = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=["src/anchor0.ts"],
    )

    report = inspect(files, surface, graph=graph)

    assert report["status"] == "candidate_possible"
    assert report["possible_candidate_paths"] == ["src/weak.ts"]
    assert report["graph_preliminary_drops"] == ["src/weak.ts"]


def test_surface_protected_file_cannot_be_candidate():
    files = [file(f"src/anchor{i}.ts", i + 1) for i in range(5)]
    files.append(file("src/weak.ts", 6, semantic=True))

    surface = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=["src/anchor0.ts"],
        references=["src/weak.ts"],
    )

    report = inspect(files, surface)

    assert report["status"] == "no_source_drop_candidate"
    assert report["possible_candidate_paths"] == []


def test_semantic_candidate_still_requires_rerank():
    files = [file(f"src/anchor{i}.ts", i + 1) for i in range(5)]
    files.append(file("src/semantic.ts", 6, semantic=True))

    surface = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=["src/anchor0.ts"],
    )

    report = inspect(files, surface)

    assert report["status"] == "candidate_possible"
    assert report["possible_candidate_paths"] == ["src/semantic.ts"]

    entry = next(x for x in report["candidates"] if x["path"] == "src/semantic.ts")

    assert entry["rerank_candidate_possible"] is True
    assert report["requires_shadow_validation"] is True


def test_incomplete_ab_cannot_claim_savings():
    report = {
        "status": "failed",
        "quality_passed": False,
        "gates": {
            "actual_active_consumed": False,
        },
        "metrics": {
            "token_usage_comparable": False,
            "llm_tokens_saved": None,
            "baseline_model_usage": {
                "usage_complete": False,
            },
            "active_model_usage": {
                "usage_complete": False,
            },
        },
        "failure_reasons": ["model_usage_unavailable_or_not_comparable"],
    }

    result = evaluate_d5_effectiveness(report)

    assert result["token_savings_confirmed"] is False
    assert result["llm_tokens_saved"] is None
    assert result["production_ready"] is False
