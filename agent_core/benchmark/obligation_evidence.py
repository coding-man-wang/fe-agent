from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from agent_core.context.model import (
    CodeContextItem,
    TaskContextBundle,
)

_IMPORT_SOURCE_PATTERN = re.compile(
    r"""
    from
    \s*
    ["']
    (?P<source>[^"']+)
    ["']
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class ObligationEvidenceEnrichmentResult:
    """
    Execution Obligation
    Context Enrichment Result。

    bundle:
        最终真正交给 Coding Agent
        的 Context。

    added_files:
        因 Similar Implementation
        补入的只读 Evidence。

    evicted_files:
        为满足 Context Budget
        被替换出去的低优先级 Context。
    """

    bundle: TaskContextBundle

    added_files: list[str]

    evicted_files: list[str]


class ObligationEvidenceEnricher:
    """
    Compressed Execution Task 的
    Similar Implementation
    Evidence Enrichment。

    目标：

        ExecutionPlanCompressor
        将多个 Logical Tasks
        合为一个 Execution Unit 后，

        Context 也必须具备能够支撑
        所有 Execution Obligations
        的 Repository Evidence。

    重要不变量：

        Evidence Enrichment
        !=
        Write Permission

    本模块：

        只增加 readable Code Context

    不修改：

        Change Surface
        Execution Surface
        PatchPlan
        Write Permission
    """

    SUPPORTED_SUFFIXES = {
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".vue",
    }

    # 至少有一个比较强的
    # repository pattern signal
    # 才加入 Context。
    MIN_CANDIDATE_SCORE = 20

    SHARED_PATTERN_MARKERS = (
        "request<",
        "request(",
        "params:",
        "usemock",
        "vite_api_mode",
        "mocklist",
    )

    def __init__(
        self,
        *,
        max_added_files: int = 2,
        max_file_bytes: int = 100_000,
    ) -> None:

        if max_added_files < 0:
            raise ValueError("max_added_files 必须 >= 0")

        if max_file_bytes < 1:
            raise ValueError("max_file_bytes 必须 >= 1")

        self.max_added_files = max_added_files

        self.max_file_bytes = max_file_bytes

    def enrich(
        self,
        *,
        repository_root: str | Path,
        bundle: TaskContextBundle,
    ) -> ObligationEvidenceEnrichmentResult:

        obligations = list(
            getattr(
                bundle.task,
                "execution_obligations",
                [],
            )
        )

        # 非 Compression Execution Task：
        # 完全保持原行为。
        if not obligations:
            return ObligationEvidenceEnrichmentResult(
                bundle=bundle,
                added_files=[],
                evicted_files=[],
            )

        if self.max_added_files == 0:
            return ObligationEvidenceEnrichmentResult(
                bundle=bundle,
                added_files=[],
                evicted_files=[],
            )

        root = Path(repository_root).resolve(strict=True)

        surface = bundle.change_surface or {}

        implementation_targets = self._stable_unique(
            [
                self._normalize(path)
                for path in surface.get(
                    "implementation_targets",
                    [],
                )
                if path
            ]
        )

        if not implementation_targets:
            return ObligationEvidenceEnrichmentResult(
                bundle=bundle,
                added_files=[],
                evicted_files=[],
            )

        existing_paths = {item.relative_path for item in bundle.code_context}

        candidates: list[
            tuple[
                int,
                str,
                str,
                list[str],
            ]
        ] = []

        # ==================================
        # Candidate Discovery
        #
        # 只扫描 Implementation Target
        # 同目录源码。
        #
        # 当前是 Conservative v1：
        #
        # src/api/orders.ts
        #     ↓
        # src/api/*.ts
        #
        # 不进行全仓库再次扫描。
        # ==================================

        for target in implementation_targets:

            target_path = self._resolve_file(
                root=root,
                relative_path=target,
            )

            if target_path is None:
                continue

            target_content = self._safe_read(target_path)

            if target_content is None:
                continue

            target_imports = self._extract_imports(target_content)

            try:
                siblings = list(target_path.parent.iterdir())

            except OSError:
                continue

            for candidate_path in siblings:

                if not (candidate_path.is_file()):
                    continue

                if candidate_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
                    continue

                try:
                    relative_path = (
                        candidate_path.resolve(strict=True).relative_to(root).as_posix()
                    )

                except (
                    OSError,
                    ValueError,
                ):
                    continue

                if relative_path == target:
                    continue

                if relative_path in existing_paths:
                    continue

                content = self._safe_read(candidate_path)

                if content is None:
                    continue

                score, reasons = self._score_candidate(
                    target_path=target,
                    target_content=(target_content),
                    target_imports=(target_imports),
                    candidate_path=(relative_path),
                    candidate_content=(content),
                )

                if score < self.MIN_CANDIDATE_SCORE:
                    continue

                candidates.append(
                    (
                        score,
                        relative_path,
                        content,
                        reasons,
                    )
                )

        if not candidates:
            return ObligationEvidenceEnrichmentResult(
                bundle=bundle,
                added_files=[],
                evicted_files=[],
            )

        # ==================================
        # Stable Ranking
        #
        # 高分优先；
        # 同分按路径排序，
        # 保证 deterministic。
        # ==================================

        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        selected_candidates: list[CodeContextItem] = []

        selected_paths = set(existing_paths)

        for (
            _score,
            relative_path,
            content,
            reasons,
        ) in candidates:

            if relative_path in selected_paths:
                continue

            if len(selected_candidates) >= self.max_added_files:
                break

            max_chars = bundle.budget.max_chars_per_file

            rendered_content = content[:max_chars]

            line_count = max(
                1,
                rendered_content.count("\n") + 1,
            )

            selected_candidates.append(
                CodeContextItem(
                    relative_path=(relative_path),
                    category="source",
                    language=(self._language_for(relative_path)),
                    content_mode=(
                        "full_file"
                        if (len(rendered_content) == len(content))
                        else "truncated"
                    ),
                    content=(rendered_content),
                    start_line=1,
                    end_line=(line_count),
                    relevance_reasons=(reasons),
                    content_chars=len(rendered_content),
                )
            )

            selected_paths.add(relative_path)

        (
            enriched_bundle,
            added_files,
            evicted_files,
        ) = self._merge_with_budget(
            bundle=bundle,
            added_items=(selected_candidates),
            implementation_targets=set(implementation_targets),
        )

        return ObligationEvidenceEnrichmentResult(
            bundle=(enriched_bundle),
            added_files=(added_files),
            evicted_files=(evicted_files),
        )

    # ==================================
    # Candidate Ranking
    # ==================================

    def _score_candidate(
        self,
        *,
        target_path: str,
        target_content: str,
        target_imports: set[str],
        candidate_path: str,
        candidate_content: str,
    ) -> tuple[
        int,
        list[str],
    ]:

        candidate_imports = self._extract_imports(candidate_content)

        shared_imports = sorted(target_imports & candidate_imports)

        score = len(shared_imports) * 20

        shared_patterns: list[str] = []

        target_lower = target_content.lower()

        candidate_lower = candidate_content.lower()

        for marker in self.SHARED_PATTERN_MARKERS:

            if marker in target_lower and marker in candidate_lower:
                score += 8

                shared_patterns.append(marker)

        reasons = [
            ("execution-obligation:" "similar-implementation"),
            ("similar-to=" f"{target_path}"),
        ]

        if shared_imports:
            reasons.append("shared-imports=" + ",".join(shared_imports))

        if shared_patterns:
            reasons.append("shared-patterns=" + ",".join(shared_patterns))

        reasons.append("candidate=" + candidate_path)

        return (
            score,
            reasons,
        )

    # ==================================
    # Context Budget
    # ==================================

    def _merge_with_budget(
        self,
        *,
        bundle: TaskContextBundle,
        added_items: list[CodeContextItem],
        implementation_targets: set[str],
    ) -> tuple[
        TaskContextBundle,
        list[str],
        list[str],
    ]:

        current = list(bundle.code_context)

        surface = bundle.change_surface or {}

        # Implementation Targets /
        # Dependency Context
        # 不允许为了 Similar Evidence
        # 被驱逐。
        protected_paths = {
            *implementation_targets,
            *[
                self._normalize(path)
                for path in surface.get(
                    "dependencies",
                    [],
                )
                if path
            ],
        }

        retained_added_paths: set[str] = set()

        added_files: list[str] = []

        evicted_files: list[str] = []

        for added in added_items:

            # --------------------------
            # max_files
            # --------------------------

            while len(current) >= bundle.budget.max_files:

                evict_index = self._find_evictable(
                    items=current,
                    protected_paths=(protected_paths | retained_added_paths),
                )

                if evict_index is None:
                    break

                removed = current.pop(evict_index)

                evicted_files.append(removed.relative_path)

            if len(current) >= bundle.budget.max_files:
                continue

            # --------------------------
            # max_total_chars
            # --------------------------

            while (
                self._total_chars(current) + added.content_chars
                > bundle.budget.max_total_chars
            ):

                evict_index = self._find_evictable(
                    items=current,
                    protected_paths=(protected_paths | retained_added_paths),
                )

                if evict_index is None:
                    break

                removed = current.pop(evict_index)

                evicted_files.append(removed.relative_path)

            if (
                self._total_chars(current) + added.content_chars
                > bundle.budget.max_total_chars
            ):
                continue

            current.append(added)

            retained_added_paths.add(added.relative_path)

            added_files.append(added.relative_path)

        updated_budget = bundle.budget.model_copy(
            deep=True,
            update={
                "used_files": len(current),
                "used_chars": self._total_chars(current),
            },
        )

        enriched_bundle = bundle.model_copy(
            deep=True,
            update={
                "code_context": current,
                "budget": updated_budget,
            },
        )

        return (
            enriched_bundle,
            self._stable_unique(added_files),
            self._stable_unique(evicted_files),
        )

    @staticmethod
    def _find_evictable(
        *,
        items: list[CodeContextItem],
        protected_paths: set[str],
    ) -> int | None:
        """
        Context Selection 本身已经是
        relevance ranked。

        因此从尾部开始寻找
        非 protected Context，
        优先驱逐低优先级条目。
        """

        for index in range(
            len(items) - 1,
            -1,
            -1,
        ):

            path = items[index].relative_path

            if path in protected_paths:
                continue

            return index

        return None

    def _safe_read(
        self,
        path: Path,
    ) -> str | None:
        """
        使用与 ExecutionPreparer 相同的
        Text IO 语义读取 Repository Source。

        重要：

        不使用：

            read_bytes().decode("utf-8")

        原因：

        Windows Repository 文件可能使用：

            CRLF: \\r\\n

        read_bytes().decode()
            → 保留 \\r\\n

        Path.read_text()
            → universal newline
            → 统一为 \\n

        ExecutionPreparer 同样使用
        Path.read_text()。

        因此 Context Evidence 和后续
        Freshness Validation 必须使用
        相同的 canonical text representation。
        """

        try:
            size_bytes = path.stat().st_size

        except OSError:
            return None

        if size_bytes > self.max_file_bytes:
            return None

        try:
            return path.read_text(encoding="utf-8")

        except (
            UnicodeDecodeError,
            OSError,
        ):
            return None

    @staticmethod
    def _resolve_file(
        *,
        root: Path,
        relative_path: str,
    ) -> Path | None:

        try:
            path = (root / relative_path).resolve(strict=True)

        except (
            OSError,
            RuntimeError,
        ):
            return None

        if not path.is_relative_to(root):
            return None

        if not path.is_file():
            return None

        return path

    @staticmethod
    def _extract_imports(
        content: str,
    ) -> set[str]:

        return {
            match.group("source") for match in _IMPORT_SOURCE_PATTERN.finditer(content)
        }

    @staticmethod
    def _normalize(
        path: str,
    ) -> str:

        return (
            path.replace(
                "\\",
                "/",
            )
            .strip()
            .strip("/")
        )

    @staticmethod
    def _language_for(
        path: str,
    ) -> str | None:

        suffix = Path(path).suffix.lower()

        mapping = {
            ".ts": "TypeScript",
            ".tsx": "TypeScript",
            ".js": "JavaScript",
            ".jsx": "JavaScript",
            ".vue": "Vue",
        }

        return mapping.get(suffix)

    @staticmethod
    def _total_chars(
        items: list[CodeContextItem],
    ) -> int:

        return sum(item.content_chars for item in items)

    @staticmethod
    def _stable_unique(
        values: list[str],
    ) -> list[str]:

        result: list[str] = []

        seen: set[str] = set()

        for value in values:

            if not value:
                continue

            if value in seen:
                continue

            seen.add(value)

            result.append(value)

        return result
