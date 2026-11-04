from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_core.code_intelligence.symbol_index import SymbolIndex
from agent_core.context.model import CodeContextItem, TaskContextBundle

SymbolWindowBenchmarkStatus = Literal["passed", "failed"]


class SymbolWindowBenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    required_symbol_ids: list[str] = Field(min_length=1)
    min_reduction_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    require_symbol_window: bool = True

    # 旧调用仍允许不传 SymbolIndex。
    # v1.4.6 Quality Hardening Benchmark 应设置为 True。
    require_independent_symbol_index: bool = False


class SymbolWindowBenchmarkEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    status: SymbolWindowBenchmarkStatus

    required_symbol_ids: list[str] = Field(default_factory=list)
    covered_symbol_ids: list[str] = Field(default_factory=list)
    localized_symbol_ids: list[str] = Field(default_factory=list)
    missing_symbol_ids: list[str] = Field(default_factory=list)

    symbol_coverage: float = Field(ge=0.0, le=1.0)
    source_file_chars: int = Field(ge=0)
    context_chars: int = Field(ge=0)
    reduction_ratio: float = Field(ge=0.0, le=1.0)

    budget_preserved: bool
    source_integrity_preserved: bool

    # v1.4.6 Quality Hardening
    metadata_integrity_preserved: bool
    independent_symbol_index_used: bool

    symbol_window_files: list[str] = Field(default_factory=list)
    context_modes: dict[str, str] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)


@dataclass
class _SourceAudit:
    source: str | None = None
    source_lines: list[str] = field(default_factory=list)

    # 这些行已被完整提供，不包含截断的半行。
    complete_ranges: list[tuple[int, int]] = field(default_factory=list)

    # 这些行在最终 Context 中出现过。
    # 可能包含最后一行仅提供部分字符的情况。
    visible_ranges: list[tuple[int, int]] = field(default_factory=list)

    issues: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.source is not None and not self.issues


class SymbolWindowBenchmarkEvaluator:
    """
    v1.4.6 Quality Hardening

    独立核验：
    1. Context 与当前 Repository Source 是否一致
    2. Segment 是否有序、无重叠、范围合法
    3. Semantic Chunk 的 end_line 是否真实
    4. 是否存在半行截断导致的覆盖误报
    5. Symbol Candidate 是否匹配独立 SymbolIndex
    6. Builder 上报的 covered/localized/missing 是否真实
    7. Budget 和 Reduction 是否基于真实字符数

    不负责：
        - 决定相关 Symbol
        - Graph Expansion
        - Change Surface
        - Write Permission
        - 证明跨函数依赖闭包
    """

    def evaluate(
        self,
        *,
        case: SymbolWindowBenchmarkCase,
        bundle: TaskContextBundle,
        repository_root: str | Path,
        symbol_index: SymbolIndex | None = None,
    ) -> SymbolWindowBenchmarkEvaluation:

        root = Path(repository_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("repository_root 必须是目录")

        required = set(case.required_symbol_ids)
        covered: set[str] = set()
        localized: set[str] = set()

        issues: list[str] = []
        source_integrity_preserved = True
        metadata_integrity_preserved = True

        actual_source_chars = 0
        actual_context_chars = 0

        if case.require_independent_symbol_index and symbol_index is None:
            metadata_integrity_preserved = False
            issues.append("independent_symbol_index_required")

        for item in bundle.code_context:
            audit = self._audit_source(root=root, item=item)

            actual_context_chars += len(item.content)

            if audit.source is not None:
                actual_source_chars += len(audit.source)

            if not audit.valid:
                source_integrity_preserved = False
                issues.extend(audit.issues)

            item_covered: set[str] = set()
            item_localized: set[str] = set()
            item_candidate_ids: set[str] = set()

            for candidate in item.symbol_candidates:
                symbol_id = candidate.symbol_id
                item_candidate_ids.add(symbol_id)

                # 校验 Builder 传递的 Symbol 身份。
                # 有独立 Index 时，用 Index 的范围作为判定依据，
                # 而不是信任 Candidate 自己的范围。
                authoritative = candidate

                if symbol_index is not None:
                    indexed = symbol_index.get_symbol(symbol_id)

                    if indexed is None:
                        metadata_integrity_preserved = False
                        issues.append(f"symbol_not_in_index:{symbol_id}")
                        continue

                    authoritative = indexed

                    if (
                        candidate.relative_path != indexed.relative_path
                        or candidate.start_line != indexed.start_line
                        or candidate.end_line != indexed.end_line
                        or candidate.kind != indexed.kind
                        or candidate.name != indexed.name
                    ):
                        metadata_integrity_preserved = False
                        issues.append(f"symbol_candidate_index_mismatch:{symbol_id}")

                if authoritative.relative_path != item.relative_path:
                    metadata_integrity_preserved = False
                    issues.append(f"symbol_path_mismatch:{symbol_id}")
                    continue

                if audit.source is None or not audit.valid:
                    continue

                start = authoritative.start_line
                end = authoritative.end_line

                if not self._valid_range(
                    start=start,
                    end=end,
                    total=len(audit.source_lines),
                ):
                    metadata_integrity_preserved = False
                    issues.append(f"symbol_range_invalid:{symbol_id}")
                    continue

                if self._range_fully_covered(
                    start=start,
                    end=end,
                    ranges=audit.complete_ranges,
                ):
                    item_covered.add(symbol_id)
                    continue

                if self._is_valid_localization(
                    item=item,
                    kind=authoritative.kind,
                    start=start,
                    end=end,
                    visible_ranges=audit.complete_ranges,
                ):
                    item_localized.add(symbol_id)

            # 独立计算的结果必须与 Builder 声明完全一致。
            independently_missing = item_candidate_ids - item_covered - item_localized

            reported_covered = set(item.covered_symbol_ids)
            reported_localized = set(item.localized_symbol_ids)
            reported_missing = set(item.missing_symbol_ids)

            for field_name, reported, actual in (
                ("covered", reported_covered, item_covered),
                ("localized", reported_localized, item_localized),
                ("missing", reported_missing, independently_missing),
            ):
                if reported != actual:
                    metadata_integrity_preserved = False
                    issues.append(
                        f"coverage_metadata_mismatch:"
                        f"{item.relative_path}:{field_name}:"
                        f"reported={sorted(reported)}:"
                        f"actual={sorted(actual)}"
                    )

            covered.update(item_covered)
            localized.update(item_localized)

        effective_coverage = covered | localized
        missing = required - effective_coverage

        symbol_coverage = len(required & effective_coverage) / len(required)

        reduction_ratio = (
            max(
                0.0,
                min(
                    1.0,
                    1.0 - actual_context_chars / actual_source_chars,
                ),
            )
            if actual_source_chars > 0
            else 0.0
        )

        symbol_window_files = [
            item.relative_path
            for item in bundle.code_context
            if item.content_mode == "symbol_window"
        ]

        budget = bundle.budget

        budget_preserved = (
            budget.used_files == len(bundle.code_context)
            and budget.used_files <= budget.max_files
            and budget.used_chars == actual_context_chars
            and actual_context_chars <= budget.max_total_chars
            and all(
                len(item.content) <= budget.max_chars_per_file
                for item in bundle.code_context
            )
        )

        if not budget_preserved:
            issues.append("context_budget_violation")

        if missing:
            issues.append("required_symbol_evidence_missing")

        if case.require_symbol_window and not symbol_window_files:
            issues.append("no_symbol_window_generated")

        if reduction_ratio < case.min_reduction_ratio:
            issues.append("reduction_ratio_below_target")

        if not source_integrity_preserved:
            issues.append("source_integrity_violation")

        if not metadata_integrity_preserved:
            issues.append("metadata_integrity_violation")

        passed = (
            not missing
            and budget_preserved
            and source_integrity_preserved
            and metadata_integrity_preserved
            and (not case.require_symbol_window or bool(symbol_window_files))
            and reduction_ratio >= case.min_reduction_ratio
        )

        return SymbolWindowBenchmarkEvaluation(
            case_id=case.case_id,
            status="passed" if passed else "failed",
            required_symbol_ids=sorted(required),
            covered_symbol_ids=sorted(required & covered),
            localized_symbol_ids=sorted(required & localized),
            missing_symbol_ids=sorted(missing),
            symbol_coverage=symbol_coverage,
            source_file_chars=actual_source_chars,
            context_chars=actual_context_chars,
            reduction_ratio=reduction_ratio,
            budget_preserved=budget_preserved,
            source_integrity_preserved=source_integrity_preserved,
            metadata_integrity_preserved=metadata_integrity_preserved,
            independent_symbol_index_used=(symbol_index is not None),
            symbol_window_files=symbol_window_files,
            context_modes={
                item.relative_path: item.content_mode for item in bundle.code_context
            },
            issues=sorted(set(issues)),
        )

    def _audit_source(
        self,
        *,
        root: Path,
        item: CodeContextItem,
    ) -> _SourceAudit:

        source = self._read_source(
            root=root,
            relative_path=item.relative_path,
        )

        if source is None:
            return _SourceAudit(issues=[f"source_unreadable:{item.relative_path}"])

        lines = source.split("\n")
        total = len(lines)
        audit = _SourceAudit(source=source, source_lines=lines)

        if item.content_chars != len(item.content):
            audit.issues.append(f"content_chars_mismatch:{item.relative_path}")

        if item.source_file_chars != len(source):
            audit.issues.append(f"source_file_chars_mismatch:{item.relative_path}")

        if item.content_mode == "full_file":
            if item.content != source or item.start_line != 1 or item.end_line != total:
                audit.issues.append(
                    f"full_file_content_or_range_mismatch:{item.relative_path}"
                )
            else:
                audit.complete_ranges = [(1, total)]
                audit.visible_ranges = [(1, total)]
            return audit

        if item.content_mode == "symbol_window":
            if item.start_line is not None or item.end_line is not None:
                audit.issues.append(
                    f"window_fake_contiguous_range:{item.relative_path}"
                )

            if not item.segments:
                audit.issues.append(f"window_without_segments:{item.relative_path}")
                return audit

            previous_end = 0
            extracted_parts: list[str] = []
            valid_segments = True

            for segment in item.segments:
                start = segment.start_line
                end = segment.end_line

                if not self._valid_range(
                    start=start,
                    end=end,
                    total=total,
                ):
                    valid_segments = False
                    audit.issues.append(
                        f"segment_range_invalid:" f"{item.relative_path}:{start}-{end}"
                    )
                    continue

                # Segment 必须按真实行号严格递增、不得重叠。
                if start <= previous_end:
                    valid_segments = False
                    audit.issues.append(
                        f"segment_order_or_overlap:"
                        f"{item.relative_path}:{start}-{end}"
                    )

                previous_end = max(previous_end, end)
                extracted_parts.append("\n".join(lines[start - 1 : end]))

            if not valid_segments:
                return audit

            reconstructed = "\n\n".join(extracted_parts)

            if item.content != reconstructed:
                audit.issues.append(f"window_content_mismatch:{item.relative_path}")
                return audit

            audit.complete_ranges = [
                (segment.start_line, segment.end_line) for segment in item.segments
            ]
            audit.visible_ranges = list(audit.complete_ranges)

            return audit

        if item.content_mode not in {"semantic_chunk", "truncated"}:
            audit.issues.append(f"unknown_content_mode:{item.relative_path}")
            return audit

        start = item.start_line
        end = item.end_line

        if (
            start is None
            or end is None
            or not self._valid_range(
                start=start,
                end=end,
                total=total,
            )
        ):
            audit.issues.append(f"chunk_range_invalid:{item.relative_path}")
            return audit

        if item.content_mode == "truncated" and start != 1:
            audit.issues.append(
                f"truncated_range_must_start_at_one:{item.relative_path}"
            )

        expected_range = "\n".join(lines[start - 1 : end])

        # 允许最后一行是前缀截断。
        # 但是不得允许任意文本或跳过中间源码。
        if not expected_range.startswith(item.content):
            audit.issues.append(f"chunk_source_mismatch:{item.relative_path}")
            return audit

        # 根据实际输出重新确定最后出现的行号。
        actual_end = start + item.content.count("\n")

        if actual_end != end:
            audit.issues.append(
                f"chunk_end_line_mismatch:"
                f"{item.relative_path}:reported={end}:actual={actual_end}"
            )
            return audit

        audit.visible_ranges = [(start, end)]

        # 整行都提供了才能作为完整 Symbol Coverage。
        # 最后一行只提供部分字符时，不得把该行当作完整。
        complete_end = end if item.content == expected_range else end - 1

        if complete_end >= start:
            audit.complete_ranges = [(start, complete_end)]

        return audit

    @staticmethod
    def _valid_range(
        *,
        start: int,
        end: int,
        total: int,
    ) -> bool:
        return 1 <= start <= end <= total

    @staticmethod
    def _range_fully_covered(
        *,
        start: int,
        end: int,
        ranges: list[tuple[int, int]],
    ) -> bool:
        cursor = start

        for segment_start, segment_end in sorted(ranges):
            if segment_end < cursor:
                continue

            if segment_start > cursor:
                return False

            cursor = max(cursor, segment_end + 1)

            if cursor > end:
                return True

        return False

    @staticmethod
    def _is_valid_localization(
        *,
        item: CodeContextItem,
        kind: str,
        start: int,
        end: int,
        visible_ranges: list[tuple[int, int]],
    ) -> bool:

        if kind not in {"component", "module"}:
            return False

        overlaps = any(
            segment_start <= end and segment_end >= start
            for segment_start, segment_end in visible_ranges
        )

        if not overlaps:
            return False

        if item.content_mode == "semantic_chunk":
            return True

        if item.content_mode != "symbol_window":
            return False

        # Window 模式只有明确来自 Container +
        # Semantic Evidence 的 Segment 才能声明定位。
        return any(
            "container_evidence" in segment.sources
            and "semantic_evidence" in segment.sources
            and segment.start_line <= end
            and segment.end_line >= start
            for segment in item.segments
        )

    @staticmethod
    def _read_source(
        *,
        root: Path,
        relative_path: str,
    ) -> str | None:

        normalized = relative_path.replace("\\", "/")
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

            if (
                not resolved.is_relative_to(root)
                or candidate.is_symlink()
                or not resolved.is_file()
            ):
                return None

            # 与 v1.4.6 Builder 保持相同编码约定，
            # 但由 Evaluator 独立读取、重新生成文本。
            return resolved.read_text(encoding="utf-8-sig")

        except (OSError, UnicodeError, ValueError):
            return None
