from pathlib import Path

from agent_core.benchmark.context_p3_cases import (
    build_cases,
)
from agent_core.benchmark.context_file_quality import (
    ContextFileOracle,
    ContextFileQualityEvaluator,
)
from agent_core.codebase.file_index import RepositoryFileIndexer
from agent_core.codebase.graph import (
    RepositoryDependencyGraphBuilder,
    RepositoryGraphContextExpander,
)
from agent_core.codebase.graph_evidence_filter import (
    GraphEvidenceFilter,
)
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryProfile,
)
from agent_core.context.builder import ProjectContextBuilder
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


def test_suite_cases_have_independent_oracles():
    cases = build_cases()

    assert len(cases) == 3
    assert len({case.case_id for case in cases}) == 3
    assert len({case.raw_requirement for case in cases}) == 3

    for case in cases:
        required = set(case.oracle.required_files)
        useful = set(case.oracle.useful_files)
        irrelevant = set(case.oracle.irrelevant_files)

        assert required
        assert case.required_symbol_ids

        assert not (required & useful)
        assert not (required & irrelevant)
        assert not (useful & irrelevant)

    user_case = next(case for case in cases if case.case_id == "USER-001")

    assert "src/components/UserFilters.vue" in user_case.oracle.required_files


def test_graph_only_required_dependency_counterexample(
    tmp_path: Path,
):
    main_path = tmp_path / "src" / "main.ts"
    side_effect_path = tmp_path / "src" / "sideEffect.ts"

    main_path.parent.mkdir(parents=True)

    main_path.write_text(
        "import './sideEffect'\n" "export const appReady = true\n",
        encoding="utf-8",
    )

    # 没有显式定义 Symbol，
    # 但存在真实的模块初始化副作用。
    side_effect_path.write_text(
        "document.title = 'App Ready'\n",
        encoding="utf-8",
    )

    index = RepositoryFileIndexer().build(tmp_path)

    main_record = next(
        item for item in index.files if item.relative_path == "src/main.ts"
    )

    retrieval_selection = HybridRelevantFileSelection(
        candidate_count=1,
        eligible_count=1,
        selected_count=1,
        total_selected_bytes=main_record.size_bytes,
        selected_files=[
            HybridRelevantFile(
                relative_path="src/main.ts",
                category="source",
                language="TypeScript",
                size_bytes=main_record.size_bytes,
                retrieval_score=1.0,
                retrieval_rank=1,
                policy_score=0,
                selection_score=1.0,
                semantic_hit_count=1,
            )
        ],
    )

    graph = RepositoryDependencyGraphBuilder().build(
        file_index=index,
        repository_revision="GRAPH-001",
    )

    expanded = RepositoryGraphContextExpander(
        max_files=2,
        max_graph_files=1,
        min_retrieval_files=1,
        dependency_depth=1,
        dependent_depth=1,
    ).expand(
        selection=retrieval_selection,
        index=index,
        graph=graph,
    )

    expanded_paths = {item.relative_path for item in expanded.selected_files}

    # 先证明 Graph 真实发现了依赖文件。
    assert "src/sideEffect.ts" in expanded_paths

    side_effect_item = next(
        item
        for item in expanded.selected_files
        if item.relative_path == "src/sideEffect.ts"
    )

    assert side_effect_item.symbol_candidates == []

    proposal = GraphEvidenceFilter().propose(selection=expanded)

    def build(selection):
        return ProjectContextBuilder().build(
            repository_root=tmp_path,
            requirement=RequirementSpec(
                title="初始化",
                summary="保留应用初始化行为",
            ),
            technical_spec=TechnicalSpec(
                summary="调查初始化依赖",
            ),
            task=TaskDefinition(
                task_id="GRAPH-001",
                task_type="code_change",
                title="应用初始化",
                objective="保留初始化行为",
                source_step_order=1,
            ),
            repository=RepositoryProfile(
                project_name="graph-fixture",
                root_path=str(tmp_path),
            ),
            selection=selection,
            max_files=2,
            max_total_chars=10000,
            max_chars_per_file=5000,
        )

    evaluator = ContextFileQualityEvaluator()

    oracle = ContextFileOracle(
        required_files=["src/sideEffect.ts"],
        useful_files=["src/main.ts"],
    )

    before = evaluator.evaluate(
        bundle=build(expanded),
        oracle=oracle,
    )

    after = evaluator.evaluate(
        bundle=build(proposal.selection),
        oracle=oracle,
    )

    assert before.recall == 1.0

    # 这里刻意验证当前策略的缺陷。
    assert "src/sideEffect.ts" in proposal.dropped_paths

    assert after.recall == 0.0
    assert after.missing_required == ["src/sideEffect.ts"]
