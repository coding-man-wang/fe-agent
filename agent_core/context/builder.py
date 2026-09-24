from pathlib import Path

from agent_core.codebase.model import (
    HybridRelevantFileSelection,
    RepositoryProfile,
    HybridRelevantFile,
)
from agent_core.context.model import (
    CodeContextItem,
    TaskContextBundle,
    ContextBudgetUsage,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


class ProjectContextBuilder:
    def build(
            self,
            *,
            repository_root: str | Path,
            requirement: RequirementSpec,
            technical_spec: TechnicalSpec,
            task: TaskDefinition,
            repository: RepositoryProfile,
            selection: HybridRelevantFileSelection,
            engineering_rules: list[str] | None = None,
            max_files: int = 6,
            max_total_chars: int = 30_000,
            max_chars_per_file: int = 8_000,
    ):
        if max_files < 1:
            raise ValueError("max_files 必须 >= 1")

        if max_total_chars < 1:
            raise ValueError("max_total_chars 必须 >= 1")

        if max_chars_per_file < 1:
            raise ValueError("max_chars_per_file 必须 >= 1")

        root = Path(repository_root).resolve()

        if not root.exists():
            raise ValueError(f"仓库目录不存在: {root}")

        if not root.is_dir():
            raise ValueError(f"仓库路径不是目录: {root}")

        code_context: list[CodeContextItem] = []

        skipped_files: list[str] = []

        compressed_files: list[str] = []

        used_chars = 0
        for relevant_file in selection.selected_files:
            if len(code_context) >= max_files:
                skipped_files.append(relevant_file.relative_path)
                continue

            remaining_chars = max_total_chars - used_chars

            if remaining_chars <= 0:
                skipped_files.append(relevant_file.relative_path)
                continue

            item = self._build_code_context_item(
                root=root,
                relevant_file=relevant_file,
                max_chars=min(
                    max_chars_per_file,
                    remaining_chars,
                ),
            )

            if item is None:
                skipped_files.append(relevant_file.relative_path)
                continue

            if item.content_mode != "full_file":
                compressed_files.append(item.relative_path)

            code_context.append(item)

            used_chars += item.content_chars
        return TaskContextBundle(
            requirement=requirement,
            technical_spec=technical_spec,
            task=task,
            repository=repository,
            engineering_rules=(engineering_rules or []),
            code_context=code_context,
            budget=ContextBudgetUsage(
                max_files=max_files,
                max_total_chars=(max_total_chars),
                max_chars_per_file=(max_chars_per_file),
                used_files=len(code_context),
                used_chars=used_chars,
                skipped_files=(skipped_files),
                compressed_files=(compressed_files),
            ),
        )

    def _build_code_context_item(
            self, *, root: Path, relevant_file: HybridRelevantFile, max_chars: int
    ) -> CodeContextItem:
        path = root / relevant_file.relative_path

        if not path.is_file():
            return None

        try:
            content = path.read_text(encoding="utf-8")

        except (
                UnicodeDecodeError,
                OSError,
        ):
            return None

        if len(content) <= max_chars:
            return CodeContextItem(
                relative_path=relevant_file.relative_path,
                category=relevant_file.category,
                language=relevant_file.language,
                content_mode="full_file",
                content=content,
                start_line=1,
                end_line=len(content.splitlines()) or 1,
                relevance_reasons=relevant_file.reasons,
                content_chars=len(content),
            )

        semantic_chunk = relevant_file.best_semantic_chunk

        if semantic_chunk is not None:
            chunk_content = semantic_chunk.content[:max_chars]

            return CodeContextItem(
                relative_path=relevant_file.relative_path,
                category=relevant_file.category,
                language=relevant_file.language,
                content_mode="semantic_chunk",
                content=chunk_content,
                start_line=semantic_chunk.start_line,
                end_line=semantic_chunk.end_line,
                relevance_reasons=relevant_file.reasons,
                content_chars=len(chunk_content),
            )

        truncated = content[:max_chars]

        return CodeContextItem(
            relative_path=(relevant_file.relative_path),
            category=(relevant_file.category),
            language=(relevant_file.language),
            content_mode="truncated",
            content=truncated,
            start_line=1,
            end_line=None,
            relevance_reasons=(relevant_file.reasons),
            content_chars=len(truncated),
        )
