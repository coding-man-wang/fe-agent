import argparse
import asyncio
import json
from pathlib import Path
from copy import deepcopy
from uuid import uuid4

from dotenv import load_dotenv

import os

import hashlib

from agent_core.benchmark.case_loader import BenchmarkCaseLoader
from agent_core.benchmark.full_execution_active_preflight import (
    ActivePreflightDecision,
    evaluate_active_preflight,
)
from apps.agent_cli.benchmark_multi_task_v1_4_8 import (
    git_command,
)

from agent_core.benchmark.full_execution_context_shadow import (
    FullExecutionContextShadowObserver,
)
from agent_core.benchmark.selection_shadow_integration import (
    ContextOptimizationShadowRunner,
)
from agent_core.benchmark.task_rerank_diagnostics import (
    TaskRerankDiagnosticRunner,
)
from agent_core.codebase.openai_reranker import (
    OpenAICompatibleReranker,
)
from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.benchmark.change_surface_evaluator import ChangeSurfaceEvaluator
from agent_core.benchmark.full_execution import FullExecutionBenchmarkService
from agent_core.benchmark.full_execution_ab import compare_full_execution_ab
from agent_core.benchmark.model_usage import MeteredLLM
from agent_core.benchmark.selection_evidence_binding import json_digest
from agent_core.llm.retrying import RetryingLLM
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

    parser.add_argument(
        "--context-shadow",
        action="store_true",
        help=(
            "在 Full Execution 的 Coding Agent 消费前"
            "执行一次只读 Context Shadow 观察"
        ),
    )

    parser.add_argument(
        "--context-active-preflight",
        action="store_true",
        help=("只检查受控 Active 实验的准入条件，" "不调用模型、不执行 Full Execution"),
    )

    parser.add_argument(
        "--context-guarded-preview",
        action="store_true",
        help=(
            "在真实 Full Execution 中校验 Guarded Context 候选；"
            "Coding Agent 仍消费 Baseline，生产 Active 保持关闭"
        ),
    )

    parser.add_argument(
        "--context-guarded-active",
        action="store_true",
        help="仅在隔离 Benchmark 中为 CASE-006/TASK-002 启用受控候选消费，生产默认保持关闭",
    )
    parser.add_argument(
        "--context-ab",
        action="store_true",
        help="固定 Prepared Input，分别运行 Baseline 和受控 Active 并输出完整 A/B 验收报告",
    )
    parser.add_argument(
        "--benchmark-repository",
        type=Path,
        default=Path(r"D:\project\fe-agent-benchmark-web"),
        help="Benchmark 目标仓库路径",
    )

    parser.add_argument(
        "--active-task-id",
        default=None,
        help="本次申请实验的 Task ID",
    )

    parser.add_argument(
        "--approved-revision",
        default=None,
        help="明确批准的目标仓库初始 Commit SHA",
    )

    return parser.parse_args()


def _checked_active_preflight(
    *,
    args,
    fe_agent_root: Path,
    benchmark_repository: Path,
) -> ActivePreflightDecision:
    """
    纯前置检查执行路径。

    禁止初始化 LLM、Retrieval、Worktree、
    FullExecutionBenchmarkService。
    """
    if args.context_shadow or args.refresh_upstream:
        raise SystemExit(
            "Active Preflight 不能与 "
            "--context-shadow / --refresh-upstream "
            "同时使用"
        )

    if not args.active_task_id:
        raise SystemExit("必须显式提供 --active-task-id")

    if not args.approved_revision:
        raise SystemExit("必须显式提供 --approved-revision")

    case = BenchmarkCaseLoader(fe_agent_root / "benchmark").load(args.case_id)

    current_revision = git_command(
        benchmark_repository,
        "rev-parse",
        "HEAD",
    )

    repository_dirty = bool(
        git_command(
            benchmark_repository,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
    )

    # 只读取既有 Canary，不在 Preflight 中重新调用模型。
    canary_path = fe_agent_root / "var" / "canary" / "context_policy_v1_4_9.json"

    canary = None
    canary_sha256 = None

    try:
        raw = canary_path.read_bytes()

        canary_sha256 = hashlib.sha256(raw).hexdigest()

        canary = json.loads(raw.decode("utf-8"))

    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        # 缺失、格式损坏均由 Gate 拒绝。
        pass

    result = evaluate_active_preflight(
        case=case,
        task_id=args.active_task_id,
        approved_revision=args.approved_revision,
        current_revision=current_revision,
        repository_dirty=repository_dirty,
        canary_report=canary,
        canary_sha256=canary_sha256,
    )

    print(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )

    if result.status != "passed":
        raise SystemExit(2)

    return result


def _run_active_preflight(
    *,
    args,
    fe_agent_root: Path,
    benchmark_repository: Path,
) -> None:
    _checked_active_preflight(
        args=args,
        fe_agent_root=fe_agent_root,
        benchmark_repository=benchmark_repository,
    )


def create_full_execution_shadow_observer():
    """
    只构造 Rerank 诊断客户端。

    Retrieval 仍由既有 FullExecutionBenchmarkService
    负责，Shadow 不会再建立自己的 Selection Service。
    """
    required = (
        "RERANK_MODEL_ID",
        "RERANK_API_KEY",
        "RERANK_BASE_URL",
    )

    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise RuntimeError("缺少 Shadow 配置: " + ", ".join(missing))

    reranker = OpenAICompatibleReranker(
        model=os.environ["RERANK_MODEL_ID"],
        api_key=os.environ["RERANK_API_KEY"],
        base_url=os.environ["RERANK_BASE_URL"],
        request_timeout_seconds=60,
    )

    runner = ContextOptimizationShadowRunner(
        rerank_runner=TaskRerankDiagnosticRunner(
            reranker=reranker,
            max_document_chars=6000,
        )
    )

    return FullExecutionContextShadowObserver(
        runner=runner,
        timeout_seconds=90.0,
    )


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


def _meter_llm(llm):
    # Meter actual provider attempts, including retries, without mutating the
    # shared runner or changing its retry policy.
    if isinstance(llm, RetryingLLM):
        wrapped, meter = _meter_llm(llm.llm)
        return RetryingLLM(llm=wrapped, retry_policy=deepcopy(llm.retry_policy)), meter
    meter = MeteredLLM(llm)
    return meter, meter


def _experiment_identity(*, case, prepared, baseline, fe_agent_root, preflight):
    artifact_root = (
        fe_agent_root
        / ".fe-agent"
        / "benchmark"
        / "harness"
        / prepared.baseline_run_id
        / "artifacts"
    )
    artifacts = {
        path.relative_to(artifact_root)
        .as_posix(): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in sorted(artifact_root.glob("*/*.json"))
    }
    if not artifacts:
        raise RuntimeError("Active experiment requires frozen upstream artifacts")
    return {
        "case_sha256": json_digest(case.model_dump(mode="json")),
        "prepared_input_sha256": json_digest(
            {
                "prepared": prepared.model_dump(mode="json"),
                "requirement": baseline.requirement_spec.model_dump(mode="json"),
                "technical_spec": baseline.technical_spec.model_dump(mode="json"),
                "task_plan": baseline.task_plan.model_dump(mode="json"),
                "project": baseline.project_context.model_dump(mode="json"),
            }
        ),
        "upstream_artifacts_sha256": json_digest(artifacts),
        "model_config_sha256": json_digest(
            {
                "models": {
                    key: os.getenv(key)
                    for key in (
                        "LLM_MODEL_ID",
                        "LLM_BASE_URL",
                        "EMBEDDING_MODEL_ID",
                        "EMBEDDING_BASE_URL",
                        "RERANK_MODEL_ID",
                        "RERANK_BASE_URL",
                    )
                },
                "implementation": "v1.4.9-final",
                "mode": "frozen_prepared_input",
            }
        ),
        "canary_sha256": preflight.canary_sha256,
        "approved_revision": preflight.approved_revision,
    }


async def _execute_prepared_run(
    *,
    fe_agent_root,
    benchmark_repository,
    runner,
    case,
    prepared,
    baseline,
    mode="baseline",
    preflight=None,
    identity=None,
    observer=None,
):
    llm, meter = _meter_llm(runner.requirement_analyzer.llm)
    repository = _repository_profile(
        baseline=baseline, benchmark_repository=benchmark_repository
    )
    coordinator = _coordinator_for_baseline(
        fe_agent_root=fe_agent_root,
        baseline_run_id=prepared.baseline_run_id,
        llm=llm,
    )
    selection_service = _build_context_selection_service(runner=runner)
    if observer is None and mode != "baseline":
        observer = create_full_execution_shadow_observer()
    return await FullExecutionBenchmarkService(
        fe_agent_root=fe_agent_root,
        benchmark_repository=benchmark_repository,
        baseline_run_id=prepared.baseline_run_id,
        llm=llm,
        coordinator=coordinator,
        context_selection_service=selection_service,
        context_shadow_observer=observer,
        context_guarded_preflight=(
            preflight if mode in {"preview", "active_experiment"} else None
        ),
        experimental_context_active=mode == "active_experiment",
        context_experiment_identity=identity,
        model_usage_meter=meter,
    ).run(
        case=deepcopy(case),
        task_plan=deepcopy(baseline.task_plan),
        requirement=deepcopy(baseline.requirement_spec),
        technical_spec=deepcopy(baseline.technical_spec),
        repository=repository,
        engineering_rules=list(baseline.project_context.engineering_rules),
        task_surfaces=deepcopy(prepared.task_surfaces),
    )


def _execution_failure_details(record):
    """Expose the failing gate without requiring another paid model run."""

    def excerpt(value):
        if not isinstance(value, str):
            return value
        return (
            value
            if len(value) <= 6000
            else value[:6000] + "\n[truncated; see run record]"
        )

    evaluation = getattr(record, "evaluation", None)
    usage = getattr(record, "model_usage", None) or {}
    return {
        "run_status": record.status,
        "run_error": excerpt(getattr(record, "error", None)),
        "failed_tasks": [
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "error": excerpt(task.error),
                "failed_checks": [
                    {
                        "command_id": check.command_id,
                        "name": check.name,
                        "details": excerpt(check.details),
                    }
                    for check in task.verification_checks
                    if check.status == "failed"
                ],
            }
            for task in getattr(record, "tasks", [])
            if task.status == "failed"
        ],
        "evaluation_status": getattr(evaluation, "status", None),
        "evaluation_failure_reasons": list(getattr(evaluation, "failure_reasons", [])),
        "hidden_acceptance_status": getattr(
            getattr(record, "hidden_acceptance", None), "status", None
        ),
        "usage_complete": usage.get("usage_complete", False),
        "failed_model_calls": usage.get("failed_calls"),
    }


def _write_ab_report(fe_agent_root, report):
    directory = fe_agent_root / ".fe-agent" / "benchmark" / "full_execution" / "ab"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (uuid4().hex + ".json")
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"ab_report={path}")
    return path


async def main():
    args = _parse_args()
    fe_agent_root = Path(__file__).resolve().parents[2]
    benchmark_repository = getattr(
        args, "benchmark_repository", Path(r"D:\project\fe-agent-benchmark-web")
    )
    case_id = args.case_id
    guarded_preview = getattr(args, "context_guarded_preview", False)
    guarded_active = getattr(args, "context_guarded_active", False)
    context_ab = getattr(args, "context_ab", False)
    guarded_mode = guarded_preview or guarded_active or context_ab
    modes = (
        args.context_active_preflight,
        args.context_shadow,
        guarded_preview,
        guarded_active,
        context_ab,
    )
    if sum(bool(value) for value in modes) > 1 or (
        guarded_mode and args.refresh_upstream
    ):
        raise SystemExit(
            "Context 模式不能与其他模式同时使用；受控实验不能使用 --refresh-upstream"
        )

    # ======================================
    # v1.4.9-P4-B-D2-A
    # Active Preflight 专用执行分支
    # ======================================
    if args.context_active_preflight:
        _run_active_preflight(
            args=args,
            fe_agent_root=fe_agent_root,
            benchmark_repository=benchmark_repository,
        )
        return

    # 防止普通 Full Execution 意外接受
    # Active 的授权参数却不执行准入检查。
    context_guarded_preflight = None
    if guarded_mode:
        # 在模型、检索服务及 Worktree 初始化前执行既有 D2-A 门禁。
        context_guarded_preflight = _checked_active_preflight(
            args=args,
            fe_agent_root=fe_agent_root,
            benchmark_repository=benchmark_repository,
        )
    elif args.active_task_id is not None or args.approved_revision is not None:
        raise SystemExit(
            "Active 参数只能与 --context-active-preflight / "
            "--context-guarded-preview / --context-guarded-active / --context-ab 一起使用"
        )

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
        if guarded_mode:
            raise SystemExit(
                "Guarded Preview 需要既有 passed Prepared Execution Input；"
                "请先完成普通 Benchmark 的上游准备"
            )
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

    common = dict(
        fe_agent_root=fe_agent_root,
        benchmark_repository=benchmark_repository,
        runner=runner,
        case=case,
        prepared=prepared,
        baseline=baseline,
    )
    identity_args = dict(
        case=case,
        prepared=prepared,
        baseline=baseline,
        fe_agent_root=fe_agent_root,
        preflight=context_guarded_preflight,
    )
    identity = (
        _experiment_identity(**identity_args)
        if (guarded_active or context_ab)
        else None
    )
    print("=== Full Execution Benchmark ===")
    if context_ab:
        try:
            active_observer = create_full_execution_shadow_observer()
            baseline_execution = await _execute_prepared_run(
                **common, mode="baseline", identity=identity
            )
        except Exception as exc:
            _write_ab_report(
                fe_agent_root,
                {
                    "version": "v1.4.9-final",
                    "status": "failed",
                    "reason": "baseline_execution_unavailable",
                    "error_type": type(exc).__name__,
                    "baseline_run_id": None,
                    "active_run_id": None,
                    "production_ready": False,
                },
            )
            raise SystemExit(2) from exc
        if (
            baseline_execution.status != "completed"
            or baseline_execution.evaluation is None
            or baseline_execution.evaluation.status != "passed"
        ):
            _write_ab_report(
                fe_agent_root,
                {
                    "version": "v1.4.9-final",
                    "status": "failed",
                    "reason": "baseline_failed",
                    "baseline_run_id": baseline_execution.run_id,
                    "baseline_record": str(
                        fe_agent_root
                        / ".fe-agent"
                        / "benchmark"
                        / "full_execution"
                        / "runs"
                        / (baseline_execution.run_id + ".json")
                    ),
                    "baseline_failure": _execution_failure_details(baseline_execution),
                    "active_run_id": None,
                    "production_ready": False,
                },
            )
            raise SystemExit(2)
        try:
            refreshed = _checked_active_preflight(
                args=args,
                fe_agent_root=fe_agent_root,
                benchmark_repository=benchmark_repository,
            )
            refreshed_identity = _experiment_identity(
                **{**identity_args, "preflight": refreshed}
            )
            if refreshed_identity != identity:
                raise RuntimeError("Frozen A/B inputs changed after Baseline")
        except (Exception, SystemExit) as exc:
            _write_ab_report(
                fe_agent_root,
                {
                    "version": "v1.4.9-final",
                    "status": "failed",
                    "reason": "admission_or_inputs_changed",
                    "error_type": type(exc).__name__,
                    "baseline_run_id": baseline_execution.run_id,
                    "active_run_id": None,
                    "production_ready": False,
                },
            )
            raise SystemExit(2) from exc
        try:
            active_execution = await _execute_prepared_run(
                **common,
                mode="active_experiment",
                preflight=refreshed,
                identity=identity,
                observer=active_observer,
            )
        except Exception as exc:
            _write_ab_report(
                fe_agent_root,
                {
                    "version": "v1.4.9-final",
                    "status": "failed",
                    "reason": "active_execution_unavailable",
                    "error_type": type(exc).__name__,
                    "baseline_run_id": baseline_execution.run_id,
                    "active_run_id": None,
                    "production_ready": False,
                },
            )
            raise SystemExit(2) from exc
        report = compare_full_execution_ab(
            baseline=baseline_execution, active=active_execution
        )
        try:
            final_preflight = _checked_active_preflight(
                args=args,
                fe_agent_root=fe_agent_root,
                benchmark_repository=benchmark_repository,
            )
            inputs_unchanged = (
                _experiment_identity(**{**identity_args, "preflight": final_preflight})
                == identity
            )
        except (Exception, SystemExit):
            inputs_unchanged = False
        if not inputs_unchanged:
            report["status"] = "failed"
            report["quality_passed"] = False
            report["failure_reasons"].append("frozen_inputs_changed_during_active")
        _write_ab_report(fe_agent_root, report)
        if report["status"] != "passed":
            raise SystemExit(2)
        return
    mode = (
        "active_experiment"
        if guarded_active
        else (
            "preview"
            if guarded_preview
            else "shadow" if args.context_shadow else "baseline"
        )
    )
    execution = await _execute_prepared_run(
        **common,
        mode=mode,
        preflight=context_guarded_preflight,
        identity=identity,
    )
    print(execution.model_dump_json(indent=2))
    print(f"full_execution_run_id={execution.run_id}")
    if guarded_active:
        task = next(
            (task for task in execution.tasks if task.task_id == "TASK-002"), None
        )
        active_report = getattr(task, "context_guarded_active", None) or {}
        if not (
            active_report.get("status") == "active"
            and active_report.get("applied") is True
            and active_report.get("active_execution_enabled") is True
            and active_report.get("production_ready") is False
        ):
            raise SystemExit(
                "Active experiment was not applied; baseline fallback is not an optimization result"
            )
        final_preflight = _checked_active_preflight(
            args=args,
            fe_agent_root=fe_agent_root,
            benchmark_repository=benchmark_repository,
        )
        if (
            _experiment_identity(**{**identity_args, "preflight": final_preflight})
            != identity
        ):
            raise SystemExit("Frozen experiment inputs changed during Active")
    if guarded_mode and (
        execution.status != "completed"
        or execution.evaluation is None
        or execution.evaluation.status != "passed"
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(main())
