"""Standalone v1.6 CLI and fixed-context / adaptive / full-agentic comparison."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from agent_core.agentic_retrieval.catalog import CommittedCatalog, ReadPolicy
from agent_core.agentic_retrieval.contracts import Criterion, InvestigationScope, RetrievalBudget
from agent_core.agentic_retrieval.controller import AgenticRetrievalController, default_runtime_config
from agent_core.agentic_retrieval.evaluation import EvaluationOracle, evaluate, summarize
from agent_core.agentic_retrieval.integration import make_embedding_provider
from agent_core.agentic_retrieval.retrieval import RetrievalBackend
from agent_core.llm.openai import OpenAILLM
from agent_core.runtime.model import RuntimeConfig
from agent_core.sdd.model import TaskDefinition


async def main() -> int:
    parser = argparse.ArgumentParser(description="v1.6 read-only evidence-driven retrieval")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", help="full commit SHA; defaults to HEAD at startup")
    parser.add_argument("--allow-prefix", action="append", default=[])
    parser.add_argument("--allow-file", action="append", default=[])
    parser.add_argument("--scope", type=Path, help="trusted InvestigationScope JSON")
    parser.add_argument("--task", help="investigation objective; use --criterion for each required question")
    parser.add_argument("--criterion", action="append", default=[])
    parser.add_argument("--non-goal", action="append", default=[])
    parser.add_argument("--seed-file", action="append", default=[], help="optional reproducible initial hint; not read authorization")
    parser.add_argument("--mode", choices=["fixed_context", "adaptive", "full_agentic", "compare"], default="adaptive")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--semantic", action="store_true", help="send authorized code chunks to configured embedding provider")
    parser.add_argument("--budget", type=Path, help="RetrievalBudget JSON override")
    parser.add_argument("--runtime-config", type=Path, help="complete RuntimeConfig JSON, or partial overrides of v1.6 defaults")
    parser.add_argument("--disable-scope-preflight", action="store_true", help="diagnostic only; final evidence gate remains enabled")
    parser.add_argument("--oracle", type=Path, help="offline expected evidence JSON; read only after all model runs")
    parser.add_argument("--output", type=Path, help="new local report file; never overwrite existing files")
    parser.add_argument("--exploration-output-tokens", type=int, default=2048)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], help="opt-in; provider must support reasoning.effort")
    args = parser.parse_args()
    if args.exploration_output_tokens < 16:
        parser.error("exploration output tokens must be >= 16")
    if not 1 <= args.trials <= 10:
        parser.error("--trials must be between 1 and 10")
    if args.scope is not None:
        if args.task or args.criterion or args.non_goal:
            parser.error("use --scope or --task/--criterion/--non-goal, not both")
        scope = InvestigationScope.model_validate_json(args.scope.read_text(encoding="utf-8"))
    else:
        if not args.task:
            parser.error("--task or --scope is required")
        questions = args.criterion or [args.task]
        scope = InvestigationScope(task_id="CLI-INVESTIGATION", objective=args.task,
                                   criteria=tuple(Criterion(criterion_id=f"C{i:02d}", question=q)
                                                  for i, q in enumerate(questions, 1)),
                                   non_goals=tuple(args.non_goal))
    budget = RetrievalBudget.model_validate_json(args.budget.read_text(encoding="utf-8")) if args.budget else RetrievalBudget()
    runtime_values = default_runtime_config().model_dump()
    if args.runtime_config:
        runtime_values.update(json.loads(args.runtime_config.read_text(encoding="utf-8")))
    config = RuntimeConfig.model_validate(runtime_values)
    policy = ReadPolicy(allowed_prefixes=tuple(p.rstrip("/") for p in args.allow_prefix),
                        allowed_paths=tuple(args.allow_file))
    # Keep Oracle files out of any source authorized for retrieval/embedding.
    # This is a path-level check; the caller must not put equivalent expected
    # answers into requirements, seeds or other authorized source files.
    if args.oracle:
        root = Path(args.repo).resolve()
        oracle_path = args.oracle.resolve()
        if oracle_path.is_relative_to(root) and policy.permits(oracle_path.relative_to(root).as_posix()):
            parser.error("Oracle cannot be inside the authorized source catalog")
    if args.output:
        if args.output.resolve().is_relative_to(Path(args.repo).resolve()):
            parser.error("--output must be outside the investigated repository")
        if args.output.exists():
            parser.error("--output must not already exist")
    load_dotenv()
    setup_started = perf_counter()
    catalog = CommittedCatalog.from_git(args.repo, policy, budget, revision=args.revision)
    llm = OpenAILLM(model=os.environ["LLM_MODEL_ID"], apikey=os.environ["LLM_API_KEY"],
                    base_url=os.environ["LLM_BASE_URL"])
    task = TaskDefinition(task_id=scope.task_id, task_type="investigation", title=scope.objective,
                          objective=scope.objective, source_step_order=1,
                          investigation_questions=[c.question for c in scope.criteria])
    modes = ["fixed_context", "adaptive", "full_agentic"] if args.mode == "compare" else [args.mode]
    runs = []
    with catalog.mirror() as root:
        backend = RetrievalBackend(catalog, root, budget)
        await backend.prepare_semantic(make_embedding_provider(args.semantic))
        if args.seed_file:
            if not set(args.seed_file) <= catalog.files.keys():
                parser.error("every seed-file must be inside the authorized committed catalog")
            initial_paths = tuple(dict.fromkeys(args.seed_file))
            if len(initial_paths) > budget.max_files:
                parser.error("seed file count exceeds budget")
        else:
            async with asyncio.timeout(budget.setup_timeout_seconds):
                initial_paths = await backend.initial_candidates(task=task, raw_requirement=scope.objective)
        setup_ms = round((perf_counter() - setup_started) * 1000, 2)
        controller = AgenticRetrievalController(llm=llm, runtime_config=config,
                                                  exploration_output_tokens=args.exploration_output_tokens,
                                                  reasoning_effort=args.reasoning_effort)
        for _ in range(args.trials):
            for mode in modes:
                runs.append(await controller.run(backend=backend, scope=scope,
                                                 initial_paths=initial_paths, mode=mode,
                                                 scope_preflight=not args.disable_scope_preflight))
    # Read evaluation expectations only after every retrieval run is complete.
    oracle = EvaluationOracle.model_validate_json(args.oracle.read_text(encoding="utf-8")) if args.oracle else None
    metrics = [evaluate(run, oracle) for run in runs]
    report = {"revision": catalog.revision, "authorized_catalog_files": len(catalog.files),
              "excluded_tree_entries": catalog.excluded_count, "initial_candidates": list(initial_paths),
              "semantic_enabled": args.semantic, "setup_ms": setup_ms,
              "initial_retrieval_diagnostics": backend.initial_diagnostics,
              "usage_note": "runtime tokens exclude embedding and initial retrieval setup; no hard billing guarantee",
              "runs": [run.model_dump(mode="json") for run in runs], "metrics": metrics,
              "summary": summarize(metrics)}
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        print(f"Report written: {args.output}")
    else:
        print(encoded)
    if args.mode == "compare":
        return 0  # blocked baseline is a meaningful comparison outcome
    return 0 if all(run.runtime.status == "completed" for run in runs) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
