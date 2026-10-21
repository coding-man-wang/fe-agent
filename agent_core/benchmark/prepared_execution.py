from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.benchmark.case_model import BenchmarkCase
from agent_core.benchmark.change_surface_evaluator import ChangeSurfaceEvaluator
from agent_core.benchmark.change_surface_model import ChangeSurfaceEvaluation
from agent_core.benchmark.run_model import BenchmarkRunRecord
from agent_core.change_surface.aggregate import ChangeSurfaceAggregator
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryFileIndex,
)
from agent_core.sdd.task_graph import topological_sort_tasks


class PreparedExecutionInputError(ValueError):
    pass


class PreparedExecutionInput(BaseModel):
    """Frozen v1.0 output consumed by v1.1 execution benchmark.

    v1.1 is intentionally an execution benchmark. Once a v1.0 Change Surface has
    passed its gate, repeated v1.1 attempts must not rerun stochastic upstream
    planning/retrieval/investigation just to debug Patch/Apply/Verify behavior.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    case_id: str
    base_commit_sha: str
    baseline_run_id: str
    prepared_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: Literal["historical_passed_surface", "fresh_passed_surface"]
    surface_evaluation: ChangeSurfaceEvaluation
    task_surfaces: dict[str, ResolvedChangeSurface] = Field(default_factory=dict)
    task_selections: dict[str, HybridRelevantFileSelection] = Field(default_factory=dict)

    def validate_for_case(self, case: BenchmarkCase) -> None:
        if self.case_id != case.case_id:
            raise PreparedExecutionInputError(
                f"Prepared case_id 不一致: expected={case.case_id}, actual={self.case_id}"
            )
        if self.base_commit_sha != case.base_commit_sha:
            raise PreparedExecutionInputError(
                "Prepared base commit 与当前 Case 不一致: "
                f"expected={case.base_commit_sha}, actual={self.base_commit_sha}"
            )
        if self.surface_evaluation.status != "passed":
            raise PreparedExecutionInputError(
                "Prepared Execution Input 的 Change Surface 并未通过"
            )


class PreparedExecutionInputStore:
    def __init__(self, *, fe_agent_root: str | Path) -> None:
        self.root = Path(fe_agent_root).resolve(strict=True)
        self.directory = self.root / ".fe-agent" / "benchmark" / "prepared_execution"

    def _path(self, case_id: str) -> Path:
        safe = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in case_id
        )
        if not safe:
            raise PreparedExecutionInputError("case_id 非法")
        return self.directory / f"{safe}.json"

    def load_optional(self, *, case: BenchmarkCase) -> PreparedExecutionInput | None:
        path = self._path(case.case_id)
        if not path.is_file():
            return None
        try:
            prepared = PreparedExecutionInput.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            prepared.validate_for_case(case)
            return prepared
        except Exception as exc:
            raise PreparedExecutionInputError(
                f"Prepared Execution Input 非法: {path}"
            ) from exc

    def save(self, prepared: PreparedExecutionInput) -> Path:
        path = self._path(prepared.case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(prepared.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)
        return path

    def clear(self, *, case_id: str) -> None:
        path = self._path(case_id)
        if path.exists():
            path.unlink()


class HistoricalPreparedExecutionFinder:
    """Find the newest historical baseline whose v1.0 Surface already passed.

    This lets v1.1 reuse a previously verified upstream artifact graph instead of
    rerunning Embedding/LLM stages on every execution attempt.
    """

    def __init__(
        self,
        *,
        fe_agent_root: str | Path,
        repository_index: RepositoryFileIndex,
    ) -> None:
        self.root = Path(fe_agent_root).resolve(strict=True)
        self.repository_index = repository_index

    def find_latest_passed(
        self,
        *,
        case: BenchmarkCase,
    ) -> PreparedExecutionInput | None:
        runs_directory = self.root / ".fe-agent" / "benchmark" / "runs"
        if not runs_directory.is_dir():
            return None

        paths = sorted(
            runs_directory.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )

        for path in paths:
            try:
                record = BenchmarkRunRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except Exception:
                continue

            prepared = self._try_build(case=case, record=record)
            if prepared is not None:
                return prepared

        return None

    def _try_build(
        self,
        *,
        case: BenchmarkCase,
        record: BenchmarkRunRecord,
    ) -> PreparedExecutionInput | None:
        if record.case_id != case.case_id:
            return None
        if record.base_commit_sha != case.base_commit_sha:
            return None
        if record.status != "completed":
            return None
        if record.task_plan is None or record.retrieval_capture is None:
            return None

        artifact_root = (
            self.root
            / ".fe-agent"
            / "benchmark"
            / "harness"
            / record.run_id
            / "artifacts"
        )
        if not artifact_root.is_dir():
            return None

        store = JsonTaskArtifactStore(root=artifact_root)
        task_surfaces: dict[str, ResolvedChangeSurface] = {}
        all_surfaces: list[ResolvedChangeSurface] = []

        for task in topological_sort_tasks(record.task_plan):
            if task.task_type not in {"code_change", "test"}:
                continue
            artifact = store.latest(
                task_id=task.task_id,
                artifact_type="change_surface",
            )
            if artifact is None:
                return None
            try:
                surface = ResolvedChangeSurface.model_validate(artifact.payload)
            except Exception:
                return None
            task_surfaces[task.task_id] = surface
            all_surfaces.append(surface)

        combined = ChangeSurfaceAggregator().combine(
            task_id=case.case_id,
            surfaces=all_surfaces,
            task_plan=record.task_plan,
        )
        evaluation = ChangeSurfaceEvaluator().evaluate(
            case=case,
            surface=combined,
        )
        if evaluation.status != "passed":
            return None

        try:
            task_selections = reconstruct_selections_from_capture(
                record=record,
                repository_index=self.repository_index,
            )
        except PreparedExecutionInputError:
            return None

        required_code_tasks = {
            task.task_id
            for task in record.task_plan.tasks
            if task.task_type == "code_change"
        }
        if not required_code_tasks.issubset(task_selections):
            return None

        return PreparedExecutionInput(
            case_id=case.case_id,
            base_commit_sha=case.base_commit_sha,
            baseline_run_id=record.run_id,
            source="historical_passed_surface",
            surface_evaluation=evaluation,
            task_surfaces=task_surfaces,
            task_selections=task_selections,
        )


def reconstruct_selections_from_capture(
    *,
    record: BenchmarkRunRecord,
    repository_index: RepositoryFileIndex,
) -> dict[str, HybridRelevantFileSelection]:
    capture = record.retrieval_capture
    if capture is None:
        raise PreparedExecutionInputError("Baseline 缺少 retrieval_capture")

    file_by_path = {item.relative_path: item for item in repository_index.files}
    result: dict[str, HybridRelevantFileSelection] = {}

    for task_capture in capture.task_retrievals:
        selected: list[HybridRelevantFile] = []
        total_bytes = 0

        for rank, item in enumerate(task_capture.selected_files, start=1):
            file_record = file_by_path.get(item.relative_path)
            if file_record is None:
                raise PreparedExecutionInputError(
                    "历史 Retrieval 文件已不存在于固定 base commit: "
                    f"{item.relative_path}"
                )

            total_bytes += file_record.size_bytes
            selected.append(
                HybridRelevantFile(
                    relative_path=item.relative_path,
                    category=item.category,
                    language=file_record.language,
                    size_bytes=file_record.size_bytes,
                    retrieval_score=item.retrieval_score,
                    retrieval_rank=rank,
                    policy_score=0,
                    policy_rank=None,
                    selection_score=item.selection_score,
                    best_semantic_chunk=None,
                    best_semantic_score=item.best_semantic_score,
                    query_hit_count=item.query_hit_count,
                    lexical_hit_count=item.lexical_hit_count,
                    semantic_hit_count=item.semantic_hit_count,
                    hybrid_hit_count=item.hybrid_hit_count,
                    matched_queries=list(item.matched_queries),
                    matched_intents=list(item.matched_intents),
                    target_match_type=item.target_match_type,
                    target_match_value=item.target_match_value,
                    context_role=item.context_role,
                    role_reason=item.role_reason,
                    rerank_score=item.rerank_score,
                    rerank_rank=item.rerank_rank,
                    reasons=list(item.reasons),
                )
            )

        result[task_capture.task_id] = HybridRelevantFileSelection(
            candidate_count=task_capture.candidate_count,
            eligible_count=max(task_capture.candidate_count, len(selected)),
            selected_count=len(selected),
            total_selected_bytes=total_bytes,
            selected_files=selected,
            skipped_by_budget=[],
            skipped_non_indexable=[],
        )

    return result
