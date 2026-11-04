from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.codebase.model import (
    CodeChunk,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)
from agent_core.context.model import (
    CodeContextSegment,
)


class SymbolWindowBuildResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    relative_path: str

    usable: bool

    worthwhile: bool

    content: str = ""

    segments: list[CodeContextSegment] = Field(
        default_factory=list,
    )

    covered_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    localized_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    missing_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    issues: list[str] = Field(
        default_factory=list,
    )

    source_file_chars: int = Field(
        ge=0,
    )

    content_chars: int = Field(
        ge=0,
    )

    reduction_ratio: float = Field(
        ge=0.0,
        le=1.0,
    )


@dataclass(
    frozen=True,
    slots=True,
)
class _SourceSpan:
    start_line: int

    end_line: int

    sources: frozenset[str]

    symbol_ids: frozenset[str]

    reasons: tuple[str, ...]


class SymbolWindowBuilder:
    """
    Deterministic Symbol Window Builder。

    职责：

        SymbolCandidate
            ↓
        Repository Source Range
            ↓
        Mandatory Evidence
            +
        Optional Coverage
            ↓
        Symbol Context Window

    不负责：

        Ranking
        Candidate Admission
        Seed Eligibility
        Graph Expansion
        Write Permission
        Change Surface

    核心 Invariant：

        Smaller Context
        !=
        Weaker Evidence
    """

    CONTAINER_KINDS = {
        "component",
        "module",
    }

    IMPORT_SCAN_SUFFIXES = {
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".mjs",
        ".cjs",
        ".vue",
    }

    _IMPORT_FROM_END = re.compile(
        r"""
        \bfrom\s*
        ["'][^"']+["']
        \s*;?\s*$
        """,
        re.VERBOSE,
    )

    _SIDE_EFFECT_IMPORT_END = re.compile(
        r"""
        ^\s*
        import\s*
        ["'][^"']+["']
        \s*;?\s*$
        """,
        re.VERBOSE,
    )

    def __init__(
        self,
        *,
        context_lines: int = 2,
        max_import_lines: int = 80,
        max_import_statement_lines: int = 40,
        container_localization_ratio: float = 0.80,
        merge_gap_lines: int = 0,
        min_savings_ratio: float = 0.10,
    ) -> None:

        if context_lines < 0:
            raise ValueError("context_lines 不能小于 0")

        if max_import_lines < 0:
            raise ValueError("max_import_lines 不能小于 0")

        if max_import_statement_lines < 1:
            raise ValueError("max_import_statement_lines " "必须 >= 1")

        if not (0.0 <= container_localization_ratio <= 1.0):
            raise ValueError("container_localization_ratio " "必须位于 0~1")

        if merge_gap_lines < 0:
            raise ValueError("merge_gap_lines 不能小于 0")

        if not (0.0 <= min_savings_ratio <= 1.0):
            raise ValueError("min_savings_ratio " "必须位于 0~1")

        self.context_lines = context_lines

        self.max_import_lines = max_import_lines

        self.max_import_statement_lines = max_import_statement_lines

        self.container_localization_ratio = container_localization_ratio

        self.merge_gap_lines = merge_gap_lines

        self.min_savings_ratio = min_savings_ratio

    def build(
        self,
        *,
        relative_path: str,
        content: str,
        candidates: list[SymbolCandidate],
        semantic_chunk: CodeChunk | None,
        max_chars: int,
    ) -> SymbolWindowBuildResult:

        if max_chars < 1:
            raise ValueError("max_chars 必须 >= 1")

        lines = self._source_lines(content)

        line_count = max(
            1,
            len(lines),
        )

        source_file_chars = len(content)

        unique_candidates = self._deduplicate_candidates(candidates)

        required_spans: list[_SourceSpan] = []

        local_context_spans: list[_SourceSpan] = []

        missing_symbol_ids: set[str] = set()

        localized_symbol_ids: set[str] = set()

        issues: list[str] = []

        valid_semantic_chunk = self._valid_semantic_chunk(
            relative_path=(relative_path),
            semantic_chunk=(semantic_chunk),
            line_count=line_count,
        )

        if semantic_chunk is not None and valid_semantic_chunk is None:
            issues.append("semantic_chunk_invalid_or_mismatched")

        # ==================================
        # 1. Mandatory Symbol Evidence
        # ==================================

        for candidate in unique_candidates:
            if candidate.relative_path != relative_path:
                missing_symbol_ids.add(candidate.symbol_id)

                issues.append("candidate_path_mismatch:" f"{candidate.symbol_id}")

                continue

            if not self._valid_range(
                start_line=(candidate.start_line),
                end_line=(candidate.end_line),
                line_count=line_count,
            ):
                missing_symbol_ids.add(candidate.symbol_id)

                issues.append(
                    "candidate_range_invalid:"
                    f"{candidate.symbol_id}:"
                    f"{candidate.start_line}-"
                    f"{candidate.end_line}"
                )

                continue

            symbol_span_lines = candidate.end_line - candidate.start_line + 1

            # ==================================
            # Container Localization
            #
            # Vue Component 的 CodeSymbol
            # 默认覆盖整个 .vue 文件。
            #
            # 如果它几乎等于整个文件，
            # 则不能因为 Component 相关
            # 自动把整份文件加入 Context。
            #
            # 已有 Semantic Evidence 时，
            # 使用它定位 Container。
            # ==================================

            is_container = candidate.kind in self.CONTAINER_KINDS and (
                symbol_span_lines / line_count >= self.container_localization_ratio
            )

            if is_container and valid_semantic_chunk is not None:
                required_spans.append(
                    _SourceSpan(
                        start_line=(valid_semantic_chunk.start_line),
                        end_line=(valid_semantic_chunk.end_line),
                        sources=frozenset(
                            {
                                "container_evidence",
                                "semantic_evidence",
                            }
                        ),
                        symbol_ids=frozenset({candidate.symbol_id}),
                        reasons=(
                            "Container Symbol "
                            "使用 Semantic Evidence "
                            "定位局部真实源码范围",
                        ),
                    )
                )

                localized_symbol_ids.add(candidate.symbol_id)

                continue

            # 普通 Function / Type /
            # Variable 等必须完整覆盖。
            required_spans.append(
                _SourceSpan(
                    start_line=(candidate.start_line),
                    end_line=(candidate.end_line),
                    sources=frozenset({"symbol"}),
                    symbol_ids=frozenset({candidate.symbol_id}),
                    reasons=("Mandatory Symbol Evidence",),
                )
            )

            # 前后几行只是 Optional Context。
            if self.context_lines:
                local_context_spans.append(
                    _SourceSpan(
                        start_line=max(
                            1,
                            candidate.start_line - self.context_lines,
                        ),
                        end_line=min(
                            line_count,
                            candidate.end_line + self.context_lines,
                        ),
                        sources=frozenset({"local_context"}),
                        symbol_ids=frozenset({candidate.symbol_id}),
                        reasons=(
                            "Symbol Local Context " f"±{self.context_lines} lines",
                        ),
                    )
                )

        if not required_spans:
            return SymbolWindowBuildResult(
                relative_path=(relative_path),
                usable=False,
                worthwhile=False,
                content="",
                segments=[],
                covered_symbol_ids=[],
                localized_symbol_ids=sorted(localized_symbol_ids),
                missing_symbol_ids=sorted(missing_symbol_ids),
                issues=sorted(set(issues or ["no_valid_symbol_evidence"])),
                source_file_chars=(source_file_chars),
                content_chars=0,
                reduction_ratio=0.0,
            )

        # ==================================
        # 2. Coverage Guard
        #
        # 首先只放 Mandatory Range。
        #
        # 如果 Mandatory 本身就超过
        # Budget：
        #
        # 不能截断 Symbol。
        # 必须 Fail Close。
        # ==================================

        selected = self._merge_spans(required_spans)

        required_content = self._render_spans(
            lines=lines,
            spans=selected,
        )

        if len(required_content) > max_chars:
            missing_symbol_ids.update(
                symbol_id for span in required_spans for symbol_id in span.symbol_ids
            )

            issues.append("mandatory_symbol_evidence_exceeds_budget")

            return SymbolWindowBuildResult(
                relative_path=(relative_path),
                usable=False,
                worthwhile=False,
                content="",
                segments=[],
                covered_symbol_ids=[],
                localized_symbol_ids=sorted(localized_symbol_ids),
                missing_symbol_ids=sorted(missing_symbol_ids),
                issues=sorted(set(issues)),
                source_file_chars=(source_file_chars),
                content_chars=0,
                reduction_ratio=0.0,
            )

        # ==================================
        # 3. Import Coverage
        #
        # 第一版保守包含可容纳的
        # Static Imports。
        #
        # 暂时不做：
        # referenced-import-only slicing。
        #
        # 这是 Coverage 优先，
        # Precision 优化可进 Backlog。
        # ==================================

        for import_span in self._collect_import_spans(
            relative_path=(relative_path),
            lines=lines,
        ):
            selected = self._try_add_span(
                lines=lines,
                selected=selected,
                candidate=(import_span),
                max_chars=max_chars,
            )

        # ==================================
        # 4. Local Context
        #
        # Optional。
        #
        # Budget 不够时丢掉，
        # 不允许挤掉 Mandatory Symbol。
        # ==================================

        for context_span in local_context_spans:
            selected = self._try_add_span(
                lines=lines,
                selected=selected,
                candidate=(context_span),
                max_chars=max_chars,
            )

        selected = self._merge_spans(selected)

        rendered = self._render_spans(
            lines=lines,
            spans=selected,
        )

        # ==================================
        # 5. Final Coverage Validation
        # ==================================

        covered_symbol_ids: set[str] = set()

        for candidate in unique_candidates:
            if candidate.symbol_id in missing_symbol_ids:
                continue

            if candidate.symbol_id in localized_symbol_ids:
                continue

            covered = any(
                (
                    span.start_line <= candidate.start_line
                    and span.end_line >= candidate.end_line
                )
                for span in selected
            )

            if covered:
                covered_symbol_ids.add(candidate.symbol_id)
            else:
                missing_symbol_ids.add(candidate.symbol_id)

        segments = [
            CodeContextSegment(
                start_line=(span.start_line),
                end_line=(span.end_line),
                sources=sorted(span.sources),
                symbol_ids=sorted(span.symbol_ids),
                reasons=sorted(set(span.reasons)),
            )
            for span in selected
        ]

        content_chars = len(rendered)

        if source_file_chars > 0:
            reduction_ratio = max(
                0.0,
                min(
                    1.0,
                    (1.0 - content_chars / source_file_chars),
                ),
            )
        else:
            reduction_ratio = 0.0

        # 小文件如果只能省几行，
        # 没必要制造碎片化 Context。
        worthwhile = (
            source_file_chars > max_chars or reduction_ratio >= self.min_savings_ratio
        )

        return SymbolWindowBuildResult(
            relative_path=(relative_path),
            usable=(not missing_symbol_ids),
            worthwhile=worthwhile,
            content=rendered,
            segments=segments,
            covered_symbol_ids=sorted(covered_symbol_ids),
            localized_symbol_ids=sorted(localized_symbol_ids),
            missing_symbol_ids=sorted(missing_symbol_ids),
            issues=sorted(set(issues)),
            source_file_chars=(source_file_chars),
            content_chars=(content_chars),
            reduction_ratio=(reduction_ratio),
        )

    # ==================================
    # Import Coverage
    # ==================================

    def _collect_import_spans(
        self,
        *,
        relative_path: str,
        lines: list[str],
    ) -> list[_SourceSpan]:

        suffix = PurePosixPath(relative_path).suffix.lower()

        if suffix not in self.IMPORT_SCAN_SUFFIXES:
            return []

        spans: list[_SourceSpan] = []

        used_lines = 0

        cursor = 0

        while cursor < len(lines):
            stripped = lines[cursor].lstrip()

            if not stripped.startswith("import"):
                cursor += 1
                continue

            # dynamic import(...)
            after_import = stripped[len("import") :]

            if after_import.startswith("("):
                cursor += 1
                continue

            start = cursor
            end = cursor

            statement: list[str] = []

            while end < len(lines) and (end - start < self.max_import_statement_lines):
                statement.append(lines[end])

                text = "\n".join(statement)

                if self._IMPORT_FROM_END.search(
                    text
                ) or self._SIDE_EFFECT_IMPORT_END.search(text):
                    break

                end += 1

            if end >= len(lines):
                break

            text = "\n".join(statement)

            if not (
                self._IMPORT_FROM_END.search(text)
                or self._SIDE_EFFECT_IMPORT_END.search(text)
            ):
                cursor = start + 1
                continue

            statement_lines = end - start + 1

            if used_lines + statement_lines > self.max_import_lines:
                break

            spans.append(
                _SourceSpan(
                    start_line=(start + 1),
                    end_line=(end + 1),
                    sources=frozenset({"import_coverage"}),
                    symbol_ids=frozenset(),
                    reasons=("Import Coverage",),
                )
            )

            used_lines += statement_lines

            cursor = end + 1

        return spans

    # ==================================
    # Span Budget
    # ==================================

    def _try_add_span(
        self,
        *,
        lines: list[str],
        selected: list[_SourceSpan],
        candidate: _SourceSpan,
        max_chars: int,
    ) -> list[_SourceSpan]:

        merged = self._merge_spans(
            [
                *selected,
                candidate,
            ]
        )

        rendered = self._render_spans(
            lines=lines,
            spans=merged,
        )

        if len(rendered) <= max_chars:
            return merged

        return selected

    def _merge_spans(
        self,
        spans: list[_SourceSpan],
    ) -> list[_SourceSpan]:

        if not spans:
            return []

        ordered = sorted(
            spans,
            key=lambda span: (
                span.start_line,
                span.end_line,
                tuple(sorted(span.sources)),
                tuple(sorted(span.symbol_ids)),
            ),
        )

        result: list[_SourceSpan] = []

        current = ordered[0]

        for span in ordered[1:]:
            # merge_gap_lines = 0
            # 仍会合并相邻范围。
            if span.start_line <= (current.end_line + self.merge_gap_lines + 1):
                current = _SourceSpan(
                    start_line=min(
                        current.start_line,
                        span.start_line,
                    ),
                    end_line=max(
                        current.end_line,
                        span.end_line,
                    ),
                    sources=(current.sources | span.sources),
                    symbol_ids=(current.symbol_ids | span.symbol_ids),
                    reasons=tuple(sorted(set(current.reasons) | set(span.reasons))),
                )

                continue

            result.append(current)

            current = span

        result.append(current)

        return result

    @staticmethod
    def _render_spans(
        *,
        lines: list[str],
        spans: list[_SourceSpan],
    ) -> str:
        """
        不插入自定义注释 Marker。

        原因：
        findings.evidence 必须能够
        直接匹配 Repository Source Text。

        不连续 Segment 之间只使用空行，
        结构信息由 segments 提供。
        """

        parts = [
            "\n".join(lines[span.start_line - 1 : span.end_line]) for span in spans
        ]

        return "\n\n".join(parts)

    # ==================================
    # Validation
    # ==================================

    @staticmethod
    def _source_lines(
        content: str,
    ) -> list[str]:
        # 和 Vue Component 当前：
        #
        # content.count("\n") + 1
        #
        # 使用同一行号语义。
        lines = content.split("\n")

        return lines or [""]

    @staticmethod
    def _valid_range(
        *,
        start_line: int,
        end_line: int,
        line_count: int,
    ) -> bool:
        return 1 <= start_line <= end_line <= line_count

    def _valid_semantic_chunk(
        self,
        *,
        relative_path: str,
        semantic_chunk: CodeChunk | None,
        line_count: int,
    ) -> CodeChunk | None:

        if semantic_chunk is None:
            return None

        if semantic_chunk.relative_path != relative_path:
            return None

        if not self._valid_range(
            start_line=(semantic_chunk.start_line),
            end_line=(semantic_chunk.end_line),
            line_count=line_count,
        ):
            return None

        return semantic_chunk

    @staticmethod
    def _deduplicate_candidates(
        candidates: list[SymbolCandidate],
    ) -> list[SymbolCandidate]:
        by_id: dict[
            str,
            SymbolCandidate,
        ] = {}

        for candidate in candidates:
            previous = by_id.get(candidate.symbol_id)

            if previous is None or candidate.score > previous.score:
                by_id[candidate.symbol_id] = candidate

        return sorted(
            by_id.values(),
            key=lambda candidate: (
                candidate.relative_path,
                candidate.start_line,
                candidate.end_line,
                candidate.symbol_id,
            ),
        )
