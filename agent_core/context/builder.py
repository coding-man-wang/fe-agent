from __future__ import annotations

from pathlib import (
    Path,
    PurePosixPath,
    PureWindowsPath,
)

from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryProfile,
)
from agent_core.context.model import (
    CodeContextItem,
    ContextBudgetUsage,
    TaskContextBundle,
)
from agent_core.context.symbol_window import (
    SymbolWindowBuildResult,
    SymbolWindowBuilder,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


class SymbolWindowCoverageError(ValueError):
    """
    Symbol-aware Context 无法在预算内
    完整保留 Mandatory Symbol Evidence。

    这是 Fail Close。

    不允许：

        为了满足 Context Budget
        静默截断关键 Symbol。
    """


class ProjectContextBuilder:
    """
    Repository Context Builder。

    v1.4.6：

        File-level Selection
            ↓
        Symbol-aware Context Compression
            ↓
        TaskContextBundle

    Symbol Window 只改变：

        “读多少源码”

    不改变：

        Write Permission
        Change Surface
        Impact Scope
    """

    DEFAULT_FULL_FILE_PREFERRED_ROLES = {
        "implementation_target",
        "test",
        "policy",
    }

    def __init__(
        self,
        *,
        symbol_window_builder: SymbolWindowBuilder | None = None,
        enable_symbol_windows: bool = True,
        full_file_preferred_roles: set[str] | None = None,
        semantic_chunk_for_unmapped_symbol_mode: bool = True,
    ) -> None:

        self.symbol_window_builder = symbol_window_builder or SymbolWindowBuilder()

        self.enable_symbol_windows = enable_symbol_windows

        self.full_file_preferred_roles = set(
            (
                full_file_preferred_roles
                if (full_file_preferred_roles is not None)
                else (self.DEFAULT_FULL_FILE_PREFERRED_ROLES)
            )
        )

        self.semantic_chunk_for_unmapped_symbol_mode = (
            semantic_chunk_for_unmapped_symbol_mode
        )

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
    ) -> TaskContextBundle:

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

        symbol_window_files: list[str] = []

        used_chars = 0

        symbol_window_saved_chars = 0

        # ==================================
        # Backward Compatibility
        #
        # 只有真正包含 Symbol Evidence 的
        # Selection 才进入 Symbol-aware Mode。
        #
        # DefaultTaskContextFactory 等旧链路
        # 没有 Symbol Evidence 时：
        #
        # 保持现有行为。
        # ==================================

        symbol_aware_mode = any(
            relevant_file.symbol_candidates
            for relevant_file in selection.selected_files
        )

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
                relevant_file=(relevant_file),
                max_chars=min(
                    max_chars_per_file,
                    remaining_chars,
                ),
                symbol_aware_mode=(symbol_aware_mode),
            )

            if item is None:
                skipped_files.append(relevant_file.relative_path)

                continue

            if item.content_mode != "full_file":
                compressed_files.append(item.relative_path)

            if item.content_mode == "symbol_window":
                symbol_window_files.append(item.relative_path)

                symbol_window_saved_chars += max(
                    0,
                    (item.source_file_chars - item.content_chars),
                )

            code_context.append(item)

            used_chars += item.content_chars

        return TaskContextBundle(
            requirement=requirement,
            technical_spec=technical_spec,
            task=task,
            repository=repository,
            engineering_rules=(engineering_rules or []),
            code_context=(code_context),
            budget=ContextBudgetUsage(
                max_files=max_files,
                max_total_chars=(max_total_chars),
                max_chars_per_file=(max_chars_per_file),
                used_files=len(code_context),
                used_chars=used_chars,
                skipped_files=(skipped_files),
                compressed_files=(compressed_files),
                symbol_window_files=(symbol_window_files),
                symbol_window_saved_chars=(symbol_window_saved_chars),
            ),
        )

    def _build_code_context_item(
        self,
        *,
        root: Path,
        relevant_file: HybridRelevantFile,
        max_chars: int,
        symbol_aware_mode: bool,
    ) -> CodeContextItem | None:

        path = self._checked_path(
            root=root,
            relative_path=(relevant_file.relative_path),
        )

        if path is None:
            return None

        try:
            # RepositorySymbolSnapshotBuilder
            # 当前同样使用 utf-8-sig。
            #
            # Context Producer 与
            # Code Intelligence 使用相同
            # Canonical Text Representation。
            content = path.read_text(encoding="utf-8-sig")

        except (
            UnicodeDecodeError,
            OSError,
        ):
            return None

        source_file_chars = len(content)

        # ==================================
        # Conservative Full-file Policy
        #
        # 小型 implementation target /
        # test / policy 文件：
        #
        # 如果预算放得下，
        # 不为了省几十个字符制造碎片。
        # ==================================

        prefer_full_file = (
            relevant_file.context_role in self.full_file_preferred_roles
            and (source_file_chars <= max_chars)
        )

        if prefer_full_file:
            return self._full_file_item(
                relevant_file=(relevant_file),
                content=content,
            )

        # ==================================
        # v1.4.6 Symbol Window
        # ==================================

        if self.enable_symbol_windows and relevant_file.symbol_candidates:
            window = self.symbol_window_builder.build(
                relative_path=(relevant_file.relative_path),
                content=content,
                candidates=(relevant_file.symbol_candidates),
                semantic_chunk=(relevant_file.best_semantic_chunk),
                max_chars=max_chars,
            )

            # ==================================
            # Coverage Guard
            #
            # Symbol 已经进入 Final Retrieval，
            # 却因为 Context Budget 被静默截断，
            # 是 correctness regression。
            #
            # 因此这里 Fail Close。
            # ==================================

            if not window.usable and window.missing_symbol_ids:
                raise (
                    SymbolWindowCoverageError(
                        "Symbol Window 无法在预算内"
                        "完整保留 Mandatory Evidence: "
                        f"path="
                        f"{relevant_file.relative_path}, "
                        f"missing="
                        f"{window.missing_symbol_ids}, "
                        f"issues="
                        f"{window.issues}"
                    )
                )

            if window.usable and window.worthwhile:
                return self._symbol_window_item(
                    relevant_file=(relevant_file),
                    window=window,
                )

        # ==================================
        # Baseline File Compression
        #
        # File Retrieval baseline 必须保留。
        #
        # 但是：
        #
        # 在 Symbol-aware Mode 下，
        # 如果某个 baseline file
        # 最终没有任何 admitted Symbol，
        # 但存在 Semantic Evidence，
        #
        # 就没有必要因为文件很短
        # 自动发送整份文件。
        # ==================================

        if (
            symbol_aware_mode
            and (self.semantic_chunk_for_unmapped_symbol_mode)
            and not (relevant_file.symbol_candidates)
            and (relevant_file.best_semantic_chunk is not None)
        ):
            semantic_item = self._semantic_chunk_item(
                relevant_file=(relevant_file),
                content=content,
                max_chars=max_chars,
            )

            if semantic_item is not None:
                return semantic_item

        # ==================================
        # Legacy-compatible Fallback
        # ==================================

        if source_file_chars <= max_chars:
            return self._full_file_item(
                relevant_file=(relevant_file),
                content=content,
            )

        if relevant_file.best_semantic_chunk is not None:
            semantic_item = self._semantic_chunk_item(
                relevant_file=(relevant_file),
                content=content,
                max_chars=max_chars,
            )

            if semantic_item is not None:
                return semantic_item

        return self._truncated_item(
            relevant_file=(relevant_file),
            content=content,
            max_chars=max_chars,
        )

    # ==================================
    # Full File
    # ==================================

    def _full_file_item(
        self,
        *,
        relevant_file: HybridRelevantFile,
        content: str,
    ) -> CodeContextItem:

        line_count = max(
            1,
            len(self._source_lines(content)),
        )

        symbol_ids = sorted(
            {symbol.symbol_id for symbol in relevant_file.symbol_candidates}
        )

        return CodeContextItem(
            relative_path=(relevant_file.relative_path),
            category=(relevant_file.category),
            language=(relevant_file.language),
            content_mode="full_file",
            content=content,
            start_line=1,
            end_line=line_count,
            relevance_reasons=list(relevant_file.reasons),
            symbol_candidates=[
                symbol.model_copy(deep=True)
                for symbol in relevant_file.symbol_candidates
            ],
            content_chars=len(content),
            covered_symbol_ids=(symbol_ids),
            source_file_chars=(len(content)),
        )

    # ==================================
    # Symbol Window
    # ==================================

    def _symbol_window_item(
        self,
        *,
        relevant_file: HybridRelevantFile,
        window: SymbolWindowBuildResult,
    ) -> CodeContextItem:

        return CodeContextItem(
            relative_path=(relevant_file.relative_path),
            category=(relevant_file.category),
            language=(relevant_file.language),
            content_mode=("symbol_window"),
            content=window.content,
            # 多段 Window 不伪装成
            # 一个连续 Source Range。
            start_line=None,
            end_line=None,
            segments=[segment.model_copy(deep=True) for segment in window.segments],
            relevance_reasons=[
                *relevant_file.reasons,
                ("Symbol Window: " "deterministic source slicing"),
            ],
            symbol_candidates=[
                symbol.model_copy(deep=True)
                for symbol in relevant_file.symbol_candidates
            ],
            content_chars=(window.content_chars),
            covered_symbol_ids=list(window.covered_symbol_ids),
            localized_symbol_ids=list(window.localized_symbol_ids),
            missing_symbol_ids=list(window.missing_symbol_ids),
            window_issues=list(window.issues),
            source_file_chars=(window.source_file_chars),
        )

    # ==================================
    # Semantic Chunk
    # ==================================

    def _semantic_chunk_item(
        self,
        *,
        relevant_file: HybridRelevantFile,
        content: str,
        max_chars: int,
    ) -> CodeContextItem | None:

        chunk = relevant_file.best_semantic_chunk

        if chunk is None:
            return None

        if chunk.relative_path != relevant_file.relative_path:
            return None

        lines = self._source_lines(content)

        line_count = len(lines)

        if not (1 <= chunk.start_line <= chunk.end_line <= line_count):
            return None

        # ==================================
        # 重要：
        #
        # 不直接复用 chunk.content。
        #
        # 按当前 Repository Source Range
        # 重新切片。
        #
        # 这样 Context Producer 和
        # Integrity Consumer 使用的是
        # 同一版本源码。
        # ==================================

        (
            chunk_content,
            actual_end_line,
        ) = self._clip_line_range(
            lines=lines,
            start_line=(chunk.start_line),
            end_line=(chunk.end_line),
            max_chars=max_chars,
        )

        if not chunk_content:
            return None

        complete_end_line = self._last_fully_provided_line(
            lines=lines,
            start_line=chunk.start_line,
            end_line=actual_end_line,
            delivered_content=chunk_content,
        )

        (
            covered,
            localized,
            missing,
        ) = self._coverage_for_range(
            relevant_file=relevant_file,
            start_line=chunk.start_line,
            end_line=complete_end_line,
        )

        return CodeContextItem(
            relative_path=(relevant_file.relative_path),
            category=(relevant_file.category),
            language=(relevant_file.language),
            content_mode=("semantic_chunk"),
            content=chunk_content,
            start_line=(chunk.start_line),
            # 修复旧实现：
            # char clipping 后不能继续
            # 使用原 chunk.end_line。
            end_line=(actual_end_line),
            relevance_reasons=list(relevant_file.reasons),
            symbol_candidates=[
                symbol.model_copy(deep=True)
                for symbol in relevant_file.symbol_candidates
            ],
            content_chars=len(chunk_content),
            covered_symbol_ids=covered,
            localized_symbol_ids=(localized),
            missing_symbol_ids=missing,
            source_file_chars=len(content),
        )

    # ==================================
    # Prefix Truncation
    # ==================================

    def _truncated_item(
        self,
        *,
        relevant_file: HybridRelevantFile,
        content: str,
        max_chars: int,
    ) -> CodeContextItem:

        lines = self._source_lines(content)

        (
            truncated,
            actual_end_line,
        ) = self._clip_line_range(
            lines=lines,
            start_line=1,
            end_line=len(lines),
            max_chars=max_chars,
        )

        complete_end_line = self._last_fully_provided_line(
            lines=lines,
            start_line=1,
            end_line=actual_end_line,
            delivered_content=truncated,
        )

        (
            covered,
            localized,
            missing,
        ) = self._coverage_for_range(
            relevant_file=relevant_file,
            start_line=1,
            end_line=complete_end_line,
        )

        return CodeContextItem(
            relative_path=(relevant_file.relative_path),
            category=(relevant_file.category),
            language=(relevant_file.language),
            content_mode="truncated",
            content=truncated,
            start_line=1,
            end_line=(actual_end_line),
            relevance_reasons=list(relevant_file.reasons),
            symbol_candidates=[
                symbol.model_copy(deep=True)
                for symbol in relevant_file.symbol_candidates
            ],
            content_chars=len(truncated),
            covered_symbol_ids=covered,
            localized_symbol_ids=(localized),
            missing_symbol_ids=missing,
            source_file_chars=len(content),
        )

    # ==================================
    # Coverage Helpers
    # ==================================

    @staticmethod
    def _coverage_for_range(
        *,
        relevant_file: HybridRelevantFile,
        start_line: int,
        end_line: int,
    ) -> tuple[
        list[str],
        list[str],
        list[str],
    ]:

        covered: list[str] = []

        localized: list[str] = []

        missing: list[str] = []

        for symbol in relevant_file.symbol_candidates:
            if start_line <= symbol.start_line and end_line >= symbol.end_line:
                covered.append(symbol.symbol_id)

                continue

            overlaps = not (
                symbol.end_line < start_line or symbol.start_line > end_line
            )

            if overlaps and symbol.kind in {
                "component",
                "module",
            }:
                localized.append(symbol.symbol_id)

                continue

            missing.append(symbol.symbol_id)

        return (
            sorted(set(covered)),
            sorted(set(localized)),
            sorted(set(missing)),
        )

    # ==================================
    # Canonical Line Clipping
    # ==================================

    @staticmethod
    def _clip_line_range(
        *,
        lines: list[str],
        start_line: int,
        end_line: int,
        max_chars: int,
    ) -> tuple[
        str,
        int,
    ]:

        selected: list[str] = []

        used = 0

        actual_end = start_line - 1

        for line_number in range(
            start_line,
            end_line + 1,
        ):
            line = lines[line_number - 1]

            separator = "\n" if selected else ""

            addition = separator + line

            if used + len(addition) <= max_chars:
                selected.append(line)

                used += len(addition)

                actual_end = line_number

                continue

            # 如果第一行本身就超过预算，
            # 至少保留真实前缀。
            #
            # end_line 仍然是该真实行号。
            if not selected:
                selected.append(line[:max_chars])

                actual_end = line_number

            break

        return (
            "\n".join(selected),
            max(
                start_line,
                actual_end,
            ),
        )

    @staticmethod
    def _source_lines(
        content: str,
    ) -> list[str]:

        lines = content.split("\n")

        return lines or [""]

    # ==================================
    # Safe Repository Path
    # ==================================

    @staticmethod
    def _checked_path(
        *,
        root: Path,
        relative_path: str,
    ) -> Path | None:

        normalized = relative_path.replace(
            "\\",
            "/",
        )

        logical = PurePosixPath(normalized)

        if (
            logical.is_absolute()
            or PureWindowsPath(relative_path).drive
            or ".." in logical.parts
            or ":" in relative_path
        ):
            return None

        candidate = root.joinpath(*logical.parts)

        try:
            resolved = candidate.resolve(strict=True)

        except OSError:
            return None

        if (
            not resolved.is_relative_to(root)
            or candidate.is_symlink()
            or not resolved.is_file()
        ):
            return None

        return resolved

    @staticmethod
    def _last_fully_provided_line(
        *,
        lines: list[str],
        start_line: int,
        end_line: int,
        delivered_content: str,
    ) -> int:
        """
        最后一个被完整提供的源码行。

        end_line:
            最后实际触及的源码行号。

        fully_provided_end_line:
            最后一个完整提供的源码行号。

        二者在字符截断发生于某行内部时并不相等。
        """
        complete_range = "\n".join(lines[start_line - 1 : end_line])

        if delivered_content == complete_range:
            return end_line

        return end_line - 1
