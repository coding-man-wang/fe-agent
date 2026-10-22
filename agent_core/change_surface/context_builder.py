from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_core.artifacts.resolver import TaskArtifactResolver
from agent_core.change_surface.context_policy import RoleAwareContextPolicy
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryFileIndex,
    RepositoryProfile,
)
from agent_core.context.builder import ProjectContextBuilder
from agent_core.context.model import TaskContextBundle
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TaskPlan,
    TechnicalSpec,
)


class ResolvedTaskContextBuilder:
    """把 ResolvedChangeSurface 真正落进下游 TaskContext。"""

    def __init__(
        self,
        *,
        artifact_resolver: TaskArtifactResolver,
        base_builder: ProjectContextBuilder | None = None,
        context_policy: RoleAwareContextPolicy | None = None,
    ) -> None:
        self.artifact_resolver = artifact_resolver
        self.base_builder = base_builder or ProjectContextBuilder()
        self.context_policy = context_policy or RoleAwareContextPolicy()

    def build(
        self,
        *,
        repository_root: str | Path,
        requirement: RequirementSpec,
        technical_spec: TechnicalSpec,
        task: TaskDefinition,
        repository: RepositoryProfile,
        index: RepositoryFileIndex,
        selection: HybridRelevantFileSelection,
        surface: ResolvedChangeSurface,
        engineering_rules: list[str] | None = None,
        max_files: int = 8,
        max_total_chars: int = 40_000,
        max_chars_per_file: int = 10_000,
        task_plan: TaskPlan | None = None,
    ) -> TaskContextBundle:
        expanded = self._expand_surface_files(
            index=index,
            selection=selection,
            surface=surface,
        )
        refined = self.context_policy.apply(
            selection=expanded,
            surface=surface,
            task=task,
        )
        base = self.base_builder.build(
            repository_root=repository_root,
            requirement=requirement,
            technical_spec=technical_spec,
            task=task,
            repository=repository,
            selection=refined,
            engineering_rules=engineering_rules,
            max_files=max_files,
            max_total_chars=max_total_chars,
            max_chars_per_file=max_chars_per_file,
        )
        investigations = self.artifact_resolver.upstream_investigations(
            task=task,
            task_plan=task_plan,
        )

        executions = self.artifact_resolver.upstream_executions(
            task=task,
            task_plan=task_plan,
        )

        upstream = [item.model_dump(mode="json") for item in investigations]

        upstream.extend(item.model_dump(mode="json") for item in executions)
        return base.model_copy(
            deep=True,
            update={
                "upstream_artifacts": upstream,
                "change_surface": surface.model_dump(mode="json"),
            },
        )

    @staticmethod
    def _expand_surface_files(
        *,
        index: RepositoryFileIndex,
        selection: HybridRelevantFileSelection,
        surface: ResolvedChangeSurface,
    ) -> HybridRelevantFileSelection:
        existing = {item.relative_path: item for item in selection.selected_files}
        file_map = {item.relative_path: item for item in index.files}
        all_paths = [
            *surface.implementation_targets,
            *surface.dependencies,
            *surface.references,
            *surface.tests,
        ]
        next_rank = (
            max([item.retrieval_rank for item in selection.selected_files] or [0]) + 1
        )
        expanded = list(selection.selected_files)
        for path in all_paths:
            if path in existing:
                continue
            record = file_map.get(path)
            if record is None or not record.is_indexable:
                continue
            expanded.append(
                HybridRelevantFile(
                    relative_path=path,
                    category=record.category,
                    language=record.language,
                    size_bytes=record.size_bytes,
                    retrieval_score=0.0,
                    retrieval_rank=next_rank,
                    policy_score=0,
                    policy_rank=None,
                    selection_score=0.0,
                    matched_queries=[],
                    matched_intents=[],
                    target_match_type="none",
                    target_match_value=None,
                    reasons=["由 ResolvedChangeSurface 补入 Context"],
                )
            )
            next_rank += 1
        return selection.model_copy(
            deep=True,
            update={
                "selected_files": expanded,
                "selected_count": len(expanded),
                "total_selected_bytes": sum(item.size_bytes for item in expanded),
            },
        )
