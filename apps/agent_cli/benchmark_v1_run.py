import asyncio
from pathlib import Path

from dotenv import load_dotenv

from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.benchmark.change_surface_evaluator import ChangeSurfaceEvaluator
from agent_core.change_surface.aggregate import ChangeSurfaceAggregator
from agent_core.change_surface.investigation import InvestigationArtifactBuilder
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.change_surface.resolver import ChangeSurfaceResolver
from agent_core.codebase.model import RepositoryProfile
from agent_core.context.builder import ProjectContextBuilder
from agent_core.harness.coordinator import HarnessV1Coordinator
from agent_core.harness.trace import JsonlHarnessTraceStore
from agent_core.sdd.task_graph import topological_sort_tasks
from apps.agent_cli.benchmark_factory import create_benchmark_runner

load_dotenv()


async def main():
    fe_agent_root = Path(__file__).resolve().parents[2]
    benchmark_repository = Path(r"D:\project\fe-agent-benchmark-web")
    case_id = "CASE-006"

    runner = create_benchmark_runner(
        fe_agent_root=fe_agent_root,
        benchmark_repository=benchmark_repository,
    )

    # 先跑现有 Benchmark，保持同一套 Requirement / Planning / Retrieval 基线。
    record = await runner.run(case_id=case_id)
    if record.status != "completed":
        print("=== Benchmark Failed ===")
        print(f"run_id={record.run_id}")
        print(f"case_id={record.case_id}")
        print(f"status={record.status}")
        print(f"stage={record.stage}")
        print("error:")
        print(record.error or "<no error recorded>")

        persisted_path = (
            fe_agent_root
            / ".fe-agent"
            / "benchmark"
            / "runs"
            / f"{record.run_id}.json"
        )
        print(f"persisted_record={persisted_path}")

        raise RuntimeError(
            "Benchmark 基线阶段失败；"
            "请以上方 stage/error 为真正根因，"
            "不要根据 status=failed 本身排查 Harness。"
        )
    if (
        record.task_plan is None
        or record.requirement_spec is None
        or record.technical_spec is None
    ):
        raise RuntimeError("Benchmark Record 缺少 Task/Requirement/Technical 数据")
    if record.project_context is None:
        raise RuntimeError("Benchmark Record 缺少 ProjectContext")

    case = runner.case_loader.load(case_id)

    capture_service = runner.retrieval_capture_service
    retrieval_service = capture_service.retrieval_service
    index = capture_service.file_indexer.build(benchmark_repository)

    project = record.project_context
    repository = RepositoryProfile(
        project_name=project.project_name,
        root_path=str(benchmark_repository),
        framework=project.framework,
        language=project.language,
        build_tool=project.build_tool,
        package_manager=project.package_manager,
        state_management=project.state_management,
        test_stack=list(project.test_stack),
    )

    llm = runner.requirement_analyzer.llm
    artifact_root = (
        fe_agent_root / ".fe-agent" / "benchmark" / "harness" / record.run_id
    )
    store = JsonTaskArtifactStore(root=artifact_root / "artifacts")
    coordinator = HarnessV1Coordinator(
        store=store,
        investigation_builder=InvestigationArtifactBuilder(llm=llm),
        surface_resolver=ChangeSurfaceResolver(llm=llm),
        trace_store=JsonlHarnessTraceStore(path=artifact_root / "trace.jsonl"),
    )
    base_context_builder = ProjectContextBuilder()

    # TaskPlan 的 JSON 数组顺序不是执行顺序。
    # 所有会产生/消费 Artifact 的工作流必须严格按 depends_on DAG 执行。
    ordered_tasks = topological_sort_tasks(record.task_plan)

    print("=== Harness Task Execution Order ===")
    for task in ordered_tasks:
        print(
            f"{task.task_id} [{task.task_type}] "
            f"depends_on={list(task.depends_on)}"
        )

    surfaces: list[ResolvedChangeSurface] = []

    for task in ordered_tasks:
        if task.task_type == "verification":
            continue

        query_specs = retrieval_service.build_query_specs(
            task=task,
            raw_requirement=case.requirement,
            requirement_spec=record.requirement_spec,
        )
        selection = await retrieval_service.retrieve(
            task=task,
            index=index,
            query_specs=query_specs,
        )
        bundle = base_context_builder.build(
            repository_root=benchmark_repository,
            requirement=record.requirement_spec,
            technical_spec=record.technical_spec,
            task=task,
            repository=repository,
            selection=selection,
            engineering_rules=list(project.engineering_rules),
            max_files=8,
            max_total_chars=40_000,
            max_chars_per_file=10_000,
        )

        if task.task_type == "investigation":
            artifact = await coordinator.record_investigation(bundle=bundle)
            print(
                f"[artifact] {task.task_id} investigation recorded: "
                f"observed_files={len(artifact.observed_files)}"
            )
            continue

        surface = await coordinator.resolve_change_surface(
            bundle=bundle,
            selection=selection,
            task_plan=record.task_plan,
        )
        surfaces.append(surface)

        # 真正构造一次 Resolved Context，验证 Artifact/Surface 能进入下游上下文。
        coordinator.build_resolved_context(
            repository_root=benchmark_repository,
            requirement=record.requirement_spec,
            technical_spec=record.technical_spec,
            repository=repository,
            index=index,
            bundle=bundle,
            selection=selection,
            surface=surface,
            engineering_rules=list(project.engineering_rules),
            task_plan=record.task_plan,
        )

    # Case 级 Surface 不能对各角色分别直接做集合并集。
    # 同一个文件在不同 Task 中可能承担不同角色，例如：
    # code_change 中是 implementation_target，test Task 中是 dependency。
    # 这里通过显式 Aggregator 统一成唯一的 Case 级角色。
    combined = ChangeSurfaceAggregator().combine(
        task_id=case_id,
        surfaces=surfaces,
        task_plan=record.task_plan,
    )

    evaluation = ChangeSurfaceEvaluator().evaluate(case=case, surface=combined)

    print("=== Retrieval Evaluation ===")
    print(record.retrieval_evaluation.model_dump_json(indent=2))
    print("=== Resolved Change Surface ===")
    print(combined.model_dump_json(indent=2))
    print("=== Change Surface Evaluation ===")
    print(evaluation.model_dump_json(indent=2))
    print(f"run_id={record.run_id}")


if __name__ == "__main__":
    asyncio.run(main())
