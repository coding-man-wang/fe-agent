# FE-Agent v1.1 Production Scope Hotfix

Fixes a semantic mismatch in production-only Full Execution Benchmark.

Before:
- task-level Change Surface could contain test files in `tests`
- Coding Context rendered `implementation_targets/tests` as writable
- Full Execution only removed `tests` immediately before Scope Guard
- the model could legitimately plan a test-file patch, then Runtime rejected it

After:
- `ProductionExecutionSurfacePolicy` is applied before Coding Context is built
- test files are projected to read-only `references`
- the same execution surface is used by Context, PatchPlan Guard and Proposal Guard
- code-change tasks with no remaining production target become `skipped_no_production_change`

Validation:

```powershell
uv run pytest tests/test_v1_1_execution_surface.py -v
uv run python -m apps.agent_cli.benchmark_v1_1_run
```
