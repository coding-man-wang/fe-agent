import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.benchmark.change_surface_evaluator import ChangeSurfaceEvaluator
from agent_core.benchmark.full_execution import FullExecutionBenchmarkService
from agent_core.benchmark.prepared_execution import (
    HistoricalPreparedExecutionFinder,
    PreparedExecutionInput,
    PreparedExecutionInputStore,
)
from agent_core.benchmark.workspace_dependencies import NodeModulesWorkspaceProvisioner
from agent_core.change_surface.aggregate import ChangeSurfaceAggregator
from agent_core.change_surface.investigation import InvestigationArtifactBuilder
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.change_surface.resolver import ChangeSurfaceResolver
from agent_core.codebase.file_index import RepositoryFileIndexer
from agent_core.codebase.model import RepositoryProfile
from agent_core.context.builder import ProjectContextBuilder
from agent_core.harness.coordinator import HarnessV1Coordinator
from agent_core.harness.trace import JsonlHarnessTraceStore
from agent_core.sdd.task_graph import topological_sort_tasks
from apps.agent_cli.benchmark_factory import create_benchmark_runner
from agent_core.codebase.graph import (
    RepositoryDependencyGraphBuilder,
    RepositoryGraphContextExpander,
)

from agent_core.codebase.task_context_selection import (
    TaskContextSelectionService,
)

load_dotenv()


def _parse_args():
    parser = argparse.ArgumentParser(
        description=("FE-Agent v1.1 Full Execution Benchmark")
    )

    parser.add_argument(
        "--case-id",
        default="CASE-006",
        help=(
            "要执行的 Benchmark Case ID，"
            "例如 CASE-001 / CASE-006。"
            "默认 CASE-006。"
        ),
    )

    parser.add_argument(
        "--refresh-upstream",
        action="store_true",
        help=(
            "重新运行 Planning/Retrieval/"
            "Investigation/Change Surface。"
            "默认 v1.1 会复用已经通过 "
            "v1.0 Gate 的 "
            "Prepared Execution Input。"
        ),
    )

    return parser.parse_args()


def _repository_profile(*, baseline, benchmark_repository: Path) -> RepositoryProfile:
    project = baseline.project_context
    if project is None:
        raise RuntimeError("Baseline record 缺少 project_context")
    return RepositoryProfile(
        project_name=project.project_name,
        root_path=str(benchmark_repository),
        framework=project.framework,
        language=project.language,
        build_tool=project.build_tool,
        package_manager=project.package_manager,
        state_management=project.state_management,
        test_stack=list(project.test_stack),
    )


def _coordinator_for_baseline(*, fe_agent_root: Path, baseline_run_id: str, llm):
    harness_root = (
        fe_agent_root / ".fe-agent" / "benchmark" / "harness" / baseline_run_id
    )
    return HarnessV1Coordinator(
        store=JsonTaskArtifactStore(root=harness_root / "artifacts"),
        investigation_builder=InvestigationArtifactBuilder(llm=llm),
        surface_resolver=ChangeSurfaceResolver(llm=llm),
        # Reused Prepared Input is immutable upstream evidence for v1.1.
        # Do not append v1.1 execution trace into the historical v1.0 trace.
        trace_store=None,
    )


def _build_context_selection_service(
    *,
    runner,
) -> TaskContextSelectionService:
    """
    创建 Full Execution 使用的
    “最新代码 Context Selection”服务。

    这里直接复用 Benchmark Runner
    已经创建好的：

    - Retrieval Service
    - Embedding / Semantic Index
    - Code Chunker

    避免再初始化一套模型客户端。
    """

    capture_service = runner.retrieval_capture_service

    return TaskContextSelectionService(
        enable_symbol_retrieval=True,
        file_indexer=(capture_service.file_indexer),
        retrieval_service=(capture_service.retrieval_service),
        code_chunker=(capture_service.code_chunker),
        semantic_index=(capture_service.semantic_index),
        graph_builder=(RepositoryDependencyGraphBuilder()),
        graph_expander=(
            RepositoryGraphContextExpander(
                max_files=8,
                max_graph_files=3,
                min_retrieval_files=5,
                dependency_depth=2,
                dependent_depth=1,
            )
        ),
    )


async def _prepare_fresh(
    *, runner, case, benchmark_repository: Path, fe_agent_root: Path
):
    baseline = await runner.run(case_id=case.case_id)
    if baseline.status != "completed":
        raise RuntimeError(
            "Baseline Benchmark 未完成: "
            f"run_id={baseline.run_id}, stage={baseline.stage}, error={baseline.error}"
        )
    if not all(
        [
            baseline.task_plan,
            baseline.requirement_spec,
            baseline.technical_spec,
            baseline.project_context,
            baseline.retrieval_capture,
        ]
    ):
        raise RuntimeError("Baseline record 缺少 v1.1 所需结构化数据")

    repository = _repository_profile(
        baseline=baseline,
        benchmark_repository=benchmark_repository,
    )
    capture_service = runner.retrieval_capture_service
    retrieval_service = capture_service.retrieval_service
    index = capture_service.file_indexer.build(benchmark_repository)
    llm = runner.requirement_analyzer.llm

    harness_root = (
        fe_agent_root / ".fe-agent" / "benchmark" / "harness" / baseline.run_id
    )
    coordinator = HarnessV1Coordinator(
        store=JsonTaskArtifactStore(root=harness_root / "artifacts"),
        investigation_builder=InvestigationArtifactBuilder(llm=llm),
        surface_resolver=ChangeSurfaceResolver(llm=llm),
        trace_store=JsonlHarnessTraceStore(path=harness_root / "trace.jsonl"),
    )
    base_context_builder = ProjectContextBuilder()

    task_surfaces: dict[str, ResolvedChangeSurface] = {}
    task_selections = {}
    surfaces: list[ResolvedChangeSurface] = []

    for task in topological_sort_tasks(baseline.task_plan):
        if task.task_type == "verification":
            continue
        query_specs = retrieval_service.build_query_specs(
            task=task,
            raw_requirement=case.requirement,
            requirement_spec=baseline.requirement_spec,
        )
        selection = await retrieval_service.retrieve(
            task=task,
            index=index,
            query_specs=query_specs,
        )
        task_selections[task.task_id] = selection
        bundle = base_context_builder.build(
            repository_root=benchmark_repository,
            requirement=baseline.requirement_spec,
            technical_spec=baseline.technical_spec,
            task=task,
            repository=repository,
            selection=selection,
            engineering_rules=list(baseline.project_context.engineering_rules),
            max_files=8,
            max_total_chars=40_000,
            max_chars_per_file=10_000,
        )

        if task.task_type == "investigation":
            investigation = await coordinator.record_investigation(bundle=bundle)

            print("\n=== Fresh Investigation ===")

            print(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "title": task.title,
                        "planning_stage": (baseline.task_plan.planning_stage),
                        "responsibilities": (list(task.responsibilities)),
                        "investigation_questions": (list(task.investigation_questions)),
                        "repository_hints": (task.repository_search_hints()),
                        "observed_files": [
                            {
                                "relative_path": (item.relative_path),
                                "observed_role": (item.observed_role),
                                "confidence": (item.confidence),
                                "evidence": list(item.evidence),
                            }
                            for item in investigation.observed_files
                        ],
                        "findings": list(investigation.findings),
                        "unresolved_questions": list(
                            investigation.unresolved_questions
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

            continue

        surface = await coordinator.resolve_change_surface(
            bundle=bundle,
            selection=selection,
            task_plan=baseline.task_plan,
        )

        print("\n=== Fresh Change Surface ===")

        print(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "title": task.title,
                    "responsibilities": (list(task.responsibilities)),
                    "repository_hints": (task.repository_search_hints()),
                    "surface": (surface.model_dump(mode="json")),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        task_surfaces[task.task_id] = surface

        surfaces.append(surface)
    target_contributors: dict[
        str,
        list[dict],
    ] = {}

    for (
        task_id,
        surface,
    ) in task_surfaces.items():

        for path in surface.implementation_targets:
            target_contributors.setdefault(
                path,
                [],
            ).append(
                {
                    "task_id": task_id,
                    "task_type": next(
                        item.task_type
                        for item in baseline.task_plan.tasks
                        if (item.task_id == task_id)
                    ),
                    "rationale": list(surface.rationale),
                }
            )

    print("\n=== Change Target Contributors ===")

    print(
        json.dumps(
            target_contributors,
            ensure_ascii=False,
            indent=2,
        )
    )
    combined = ChangeSurfaceAggregator().combine(
        task_id=case.case_id,
        surfaces=surfaces,
        task_plan=baseline.task_plan,
    )
    evaluation = ChangeSurfaceEvaluator().evaluate(case=case, surface=combined)
    print("=== Change Surface Evaluation (fresh upstream) ===")
    print(evaluation.model_dump_json(indent=2))
    if evaluation.status != "passed":
        raise RuntimeError(
            "Change Surface 未通过；本次 fresh upstream 不会覆盖已冻结的 Prepared Input"
        )

    repository_aware_plan = coordinator.replan(
        task_plan=(baseline.task_plan),
        repository_revision=(baseline.base_commit_sha),
    )

    print("\n=== Repository-aware " "Task Plan ===")

    print(repository_aware_plan.model_dump_json(indent=2))
    prepared = PreparedExecutionInput(
        case_id=case.case_id,
        base_commit_sha=case.base_commit_sha,
        baseline_run_id=baseline.run_id,
        source="fresh_passed_surface",
        surface_evaluation=evaluation,
        task_surfaces=task_surfaces,
        task_selections=task_selections,
    )
    return prepared


async def main():
    args = _parse_args()
    fe_agent_root = Path(__file__).resolve().parents[2]
    benchmark_repository = Path(r"D:\project\fe-agent-benchmark-web")
    case_id = args.case_id

    dependency_preflight = NodeModulesWorkspaceProvisioner(
        dependency_source_root=benchmark_repository
    )
    dependency_preflight.require_source_ready()
    print("=== Full Execution Environment Preflight ===")
    print("node/npm: ready")
    print("node_modules verification binaries: ready")

    runner = create_benchmark_runner(
        fe_agent_root=fe_agent_root,
        benchmark_repository=benchmark_repository,
    )
    case = runner.case_loader.load(case_id)
    prepared_store = PreparedExecutionInputStore(fe_agent_root=fe_agent_root)
    repository_index = RepositoryFileIndexer().build(benchmark_repository)

    prepared = None
    if not args.refresh_upstream:
        prepared = prepared_store.load_optional(case=case)
        if prepared is None:
            prepared = HistoricalPreparedExecutionFinder(
                fe_agent_root=fe_agent_root,
                repository_index=repository_index,
            ).find_latest_passed(case=case)
            if prepared is not None:
                prepared_store.save(prepared)

    if prepared is None:
        if not args.refresh_upstream:
            print(
                "没有找到历史 passed Prepared Input；"
                "将执行一次 fresh upstream，只有通过 Gate 才会冻结。"
            )
        prepared = await _prepare_fresh(
            runner=runner,
            case=case,
            benchmark_repository=benchmark_repository,
            fe_agent_root=fe_agent_root,
        )
        prepared_store.save(prepared)

    prepared.validate_for_case(case)
    print("=== Prepared Execution Input ===")
    print(f"source={prepared.source}")
    print(f"baseline_run_id={prepared.baseline_run_id}")
    print(f"base_commit_sha={prepared.base_commit_sha}")
    print(
        "change_surface="
        f"{prepared.surface_evaluation.status}, "
        f"recall={prepared.surface_evaluation.target_recall}, "
        f"precision={prepared.surface_evaluation.target_precision}"
    )

    baseline = runner.run_store.load(prepared.baseline_run_id)
    if not all(
        [
            baseline.task_plan,
            baseline.requirement_spec,
            baseline.technical_spec,
            baseline.project_context,
        ]
    ):
        raise RuntimeError("Prepared baseline record 缺少结构化数据")

    llm = runner.requirement_analyzer.llm
    repository = _repository_profile(
        baseline=baseline,
        benchmark_repository=benchmark_repository,
    )
    coordinator = _coordinator_for_baseline(
        fe_agent_root=fe_agent_root,
        baseline_run_id=prepared.baseline_run_id,
        llm=llm,
    )

    print("=== Full Execution Benchmark ===")
    context_selection_service = _build_context_selection_service(runner=runner)

    execution = await FullExecutionBenchmarkService(
        fe_agent_root=fe_agent_root,
        benchmark_repository=(benchmark_repository),
        baseline_run_id=(prepared.baseline_run_id),
        llm=llm,
        coordinator=coordinator,
        context_selection_service=(context_selection_service),
    ).run(
        case=case,
        task_plan=baseline.task_plan,
        requirement=(baseline.requirement_spec),
        technical_spec=(baseline.technical_spec),
        repository=repository,
        engineering_rules=list(baseline.project_context.engineering_rules),
        task_surfaces=(prepared.task_surfaces),
    )
    print(execution.model_dump_json(indent=2))
    print(f"full_execution_run_id={execution.run_id}")


if __name__ == "__main__":
    asyncio.run(main())
