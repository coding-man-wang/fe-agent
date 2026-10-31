"""符号阅读证据：名称排序、一跳图扩展、保留文件基线的结果融合。"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from agent_core.code_intelligence.symbol_index import SymbolIndex
from agent_core.code_intelligence.symbol_graph import SymbolGraph
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryFileIndex,
)
from agent_core.codebase.symbol_retrieval_model import SymbolCandidate


def _tokens(text: str) -> set[str]:
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return set(re.findall(r"[^\W_]+", text.casefold()))


def _sort_key(candidate: SymbolCandidate) -> tuple:
    return (
        -candidate.score,
        candidate.source != "name",
        candidate.relative_path,
        candidate.start_line,
        candidate.symbol_id,
    )


class SymbolRanker:
    """名称排序器：精确标识符优先，再比较驼峰/下划线词元覆盖。"""

    def rank(
        self, *, queries: list[str], symbol_index: SymbolIndex, limit: int = 20
    ) -> list[SymbolCandidate]:
        if limit < 0:
            raise ValueError("limit 不能小于 0")
        queries = sorted({query.strip() for query in queries if query.strip()})
        candidates = []
        for symbol in symbol_index.all_symbols():
            name_tokens = _tokens(symbol.name)
            best_score = 0.0
            matches, reasons = [], []
            for query in queries:
                # 防止 getUser 命中 getUserList；不根据文件名猜测符号。
                exact = re.search(
                    r"(?<![\w$])" + re.escape(symbol.name) + r"(?![\w$])",
                    query,
                    re.IGNORECASE,
                )
                overlap = name_tokens & _tokens(query)
                if exact:
                    score = 1.0
                    reason = f"符号名称精确匹配: {symbol.name}"
                elif overlap:
                    score = 0.6 * len(overlap) / len(name_tokens)
                    reason = "符号名称词元匹配: " + ", ".join(sorted(overlap))
                else:
                    continue
                best_score = max(best_score, score)
                matches.append(query)
                reasons.append(reason)
            if best_score:
                candidates.append(
                    SymbolCandidate(
                        **symbol.model_dump(
                            include={
                                "symbol_id",
                                "name",
                                "kind",
                                "relative_path",
                                "start_line",
                                "end_line",
                            }
                        ),
                        score=best_score,
                        matched_queries=matches,
                        reasons=sorted(set(reasons)),
                    )
                )
        return sorted(candidates, key=_sort_key)[:limit]


class SymbolGraphExpander:
    """图扩展：仅沿已解析的调用/引用边补一跳邻居，不推导修改范围。"""

    def expand(
        self,
        *,
        candidates: list[SymbolCandidate],
        symbol_index: SymbolIndex,
        symbol_graph: SymbolGraph,
        max_neighbors: int = 20,
    ) -> list[SymbolCandidate]:
        if max_neighbors < 0:
            raise ValueError("max_neighbors 不能小于 0")
        seeds = {}
        for candidate in sorted(candidates, key=_sort_key):
            if symbol_index.get_symbol(candidate.symbol_id) is not None:
                seeds.setdefault(candidate.symbol_id, candidate)
        neighbors = {}
        for seed in seeds.values():
            # 扩展结果不能再次成为种子，确保循环和高连接度不会递归放大。
            if seed.source != "name":
                continue
            for direction, edges in (
                ("outgoing", symbol_graph.outgoing(seed.symbol_id)),
                ("incoming", symbol_graph.incoming(seed.symbol_id)),
            ):
                for edge in sorted(
                    edges,
                    key=lambda e: (
                        e.source_symbol_id,
                        e.target_symbol_id,
                        e.relation,
                        e.line_number,
                    ),
                ):
                    if edge.relation not in {"calls", "references"}:
                        continue
                    target_id = (
                        edge.target_symbol_id
                        if direction == "outgoing"
                        else edge.source_symbol_id
                    )
                    # 两个端点都必须属于当前索引，悬空边不得参与扩展。
                    if (
                        symbol_index.get_symbol(edge.source_symbol_id) is None
                        or symbol_index.get_symbol(edge.target_symbol_id) is None
                    ):
                        continue
                    if target_id in seeds:
                        continue
                    symbol = symbol_index.get_symbol(target_id)
                    candidate = SymbolCandidate(
                        **symbol.model_dump(
                            include={
                                "symbol_id",
                                "name",
                                "kind",
                                "relative_path",
                                "start_line",
                                "end_line",
                            }
                        ),
                        score=seed.score * 0.5,
                        source="graph",
                        depth=1,
                        seed_symbol_id=seed.symbol_id,
                        relation=edge.relation,
                        direction=direction,
                        matched_queries=list(seed.matched_queries),
                        reasons=[
                            f"符号图一跳 {direction}/{edge.relation}: "
                            f"{seed.symbol_id} -> {target_id}; line={edge.line_number}"
                        ],
                    )
                    previous = neighbors.get(target_id)
                    if previous is None or _sort_key(candidate) < _sort_key(previous):
                        neighbors[target_id] = candidate
        return sorted(
            [
                *seeds.values(),
                *sorted(neighbors.values(), key=_sort_key)[:max_neighbors],
            ],
            key=_sort_key,
        )


class SymbolResultFusion:
    """结果融合：原文件集合与顺序是下限，符号新文件只消费剩余预算。"""

    @staticmethod
    def _readable(root: Path, path: str) -> bool:
        normalized = path.replace("\\", "/")
        if (
            PurePosixPath(normalized).is_absolute()
            or PureWindowsPath(path).drive
            or ".." in PurePosixPath(normalized).parts
        ):
            return False
        try:
            resolved = (root / normalized).resolve(strict=True)
            return resolved.is_relative_to(root) and resolved.is_file()
        except (OSError, ValueError):
            return False

    def fuse(
        self,
        *,
        selection: HybridRelevantFileSelection,
        index: RepositoryFileIndex,
        candidates: list[SymbolCandidate],
        max_files: int,
        max_total_bytes: int,
    ) -> HybridRelevantFileSelection:
        if max_files < 0 or max_total_bytes < 0:
            raise ValueError("文件和字节预算不能小于 0")
        result = selection.model_copy(deep=True)
        records = {record.relative_path: record for record in index.files}
        existing = {file.relative_path: file for file in result.selected_files}
        known = (
            set(existing)
            | set(selection.skipped_by_budget)
            | set(selection.skipped_non_indexable)
        )
        root = Path(index.root_path).resolve()
        by_file = {}
        seen_ids = set()
        for candidate in sorted(candidates, key=_sort_key):
            if candidate.symbol_id in seen_ids:
                continue
            seen_ids.add(candidate.symbol_id)
            by_file.setdefault(candidate.relative_path, []).append(candidate)
        total_bytes = sum(file.size_bytes for file in result.selected_files)
        next_rank = (
            max((file.retrieval_rank for file in result.selected_files), default=0) + 1
        )
        for path, symbols in by_file.items():
            record = records.get(path)
            if path not in known:
                result.candidate_count += 1
            if (
                record is None
                or not record.is_indexable
                or not self._readable(root, path)
            ):
                if path not in existing:
                    result.skipped_non_indexable.append(path)
                continue
            if path not in known:
                result.eligible_count += 1
            if path in existing:
                # 不修改原检索分数、角色或目标匹配字段。
                existing[path].symbol_candidates = [
                    item.model_copy(deep=True) for item in symbols
                ]
                continue
            if (
                len(result.selected_files) >= max_files
                or total_bytes + record.size_bytes > max_total_bytes
            ):
                result.skipped_by_budget.append(path)
                continue
            file = HybridRelevantFile(
                relative_path=path,
                category=record.category,
                language=record.language,
                size_bytes=record.size_bytes,
                retrieval_score=0.0,
                retrieval_rank=next_rank,
                policy_score=0,
                selection_score=0.0,
                context_role="reference",
                role_reason="符号补充阅读证据，不授予修改权限",
                target_match_type="none",
                symbol_candidates=symbols,
                matched_queries=sorted(
                    {query for symbol in symbols for query in symbol.matched_queries}
                ),
                reasons=["符号补充候选；使用原文件链路剩余预算"],
            )
            result.selected_files.append(file)
            existing[path] = file
            next_rank += 1
            total_bytes += record.size_bytes
        result.selected_count = len(result.selected_files)
        result.total_selected_bytes = total_bytes
        result.skipped_by_budget = sorted(
            set(result.skipped_by_budget) - existing.keys()
        )
        result.skipped_non_indexable = sorted(
            set(result.skipped_non_indexable) - existing.keys()
        )
        return result
