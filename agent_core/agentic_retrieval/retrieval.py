"""Reuse FE-Agent's retrieval primitives against one verified temporary mirror."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from agent_core.codebase.exact_search import ExactCodeSearcher
from agent_core.codebase.file_index import RepositoryFileIndexer
from agent_core.codebase.graph.builder import RepositoryDependencyGraphBuilder
from agent_core.codebase.hybrid_retrieval import HybridRetriever
from agent_core.codebase.model import SemanticSearchResponse
from agent_core.codebase.semantic_search import InMemorySemanticIndex
from agent_core.codebase.chunker import CodeChunker
from agent_core.codebase.hybrid_relevance import HybridRelevantFileSelector
from agent_core.codebase.task_retrieval import TaskRetrievalService

from .catalog import CommittedCatalog
from .contracts import Contract, EvidenceGap, RetrievalBudget


class Candidate(Contract):
    path: str
    line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    channel: Literal["path", "symbol", "dependency", "exact", "hybrid"]


class NoSemantic:
    async def search(self, query, *, top_k=5):
        raise RuntimeError("semantic retrieval was not enabled")


class RetrievalBackend:
    def __init__(self, catalog: CommittedCatalog, mirror_root: Path, budget: RetrievalBudget,
                 *, aliases: dict[str, str] | None = None):
        self.catalog = catalog
        self.budget = budget
        self.index = RepositoryFileIndexer(max_indexable_file_size=budget.max_file_bytes).build(mirror_root)
        # Some repository-indexer exclusions are intentional. The search index
        # itself must not include any file outside the admitted committed catalog.
        if any(record.relative_path not in catalog.files for record in self.index.files):
            raise ValueError("retrieval index escapes authorized catalog")
        self.exact = ExactCodeSearcher()
        self.hybrid = HybridRetriever()
        self.aliases = aliases
        self._graph = None
        self._symbol_snapshot = None
        self._symbol_attempted = False
        self.semantic = None
        self.diagnostics: list[str] = []

    def reset_runtime_caches(self):
        """Reuse the embedding index, not query/graph warmup across measured runs."""
        self._graph = None
        self._symbol_snapshot = None
        self._symbol_attempted = False
        self.diagnostics.clear()
        if self.semantic is not None:
            self.semantic.query_vector_cache.clear()

    async def prepare_semantic(self, embedding_provider):
        if embedding_provider is None:
            return
        chunks = CodeChunker().build_chunks(self.index)
        if len(chunks) > self.budget.max_semantic_chunks:
            raise ValueError("semantic chunk budget exceeded; narrow authorization")
        self.semantic = InMemorySemanticIndex(embedding_provider)
        try:
            async with asyncio.timeout(self.budget.setup_timeout_seconds):
                await self.semantic.build(chunks)
        except Exception:
            self.semantic = None
            raise

    async def initial_candidates(self, *, task, requirement=None, raw_requirement=None):
        # Exactly the existing TaskRetrievalService for the standalone CLI.
        # A Harness bundle already has its own richer v1.4.9 selection; do not
        # rerun it there. Rerank is not used in the dynamic inner loop.
        service = TaskRetrievalService(
            exact_searcher=self.exact,
            semantic_searcher=self.semantic or NoSemantic(),
            hybrid_retriever=self.hybrid,
            selector=HybridRelevantFileSelector(),
            max_files=self.budget.initial_max_files,
            max_total_bytes=self.budget.max_evidence_chars * 4,
        )
        selection = await service.retrieve(task=task, index=self.index,
                                           raw_requirement=raw_requirement,
                                           requirement_spec=requirement,
                                           semantic_enabled=self.semantic is not None)
        return tuple(dict.fromkeys(item.relative_path for item in selection.selected_files
                                   if item.relative_path in self.catalog.files))

    def _dependency(self, gap: EvidenceGap) -> list[Candidate]:
        if not gap.source_path or gap.source_path not in self.catalog.files:
            return []
        if self._graph is None:
            self._graph = RepositoryDependencyGraphBuilder(aliases=self.aliases).build(
                file_index=self.index, repository_revision=self.catalog.revision)
        edges = [edge for edge in self._graph.edges if edge.source_path == gap.source_path]
        # A known module specifier narrows to its exact resolved edge. General
        # dependency discovery is explicit: query="*". Do not fuzzy-resolve it.
        if gap.query != "*":
            edges = [edge for edge in edges if gap.query in {edge.specifier, edge.target_path}]
        return [Candidate(path=edge.target_path, channel="dependency") for edge in edges]

    def _symbols(self, gap: EvidenceGap) -> list[Candidate]:
        if not self._symbol_attempted:
            self._symbol_attempted = True
            try:
                from agent_core.code_intelligence.repository_snapshot import RepositorySymbolSnapshotBuilder
                self._symbol_snapshot = RepositorySymbolSnapshotBuilder(
                    max_file_bytes=self.budget.max_file_bytes).build(
                        file_index=self.index, repository_revision=self.catalog.revision,
                        aliases=self.aliases)
                if self._symbol_snapshot.issues:
                    self.diagnostics.append("symbol_snapshot_partial")
            except Exception:
                # Symbol channel is optional on repositories without the parser.
                # Its absence is observable; the exact/graph channels can still work.
                self.diagnostics.append("symbol_snapshot_unavailable")
        if self._symbol_snapshot is None:
            return []
        symbols = self._symbol_snapshot.symbol_index.find_by_name(gap.query)
        neighbors = set()
        if gap.source_path:
            dep_gap = gap.model_copy(update={"query": "*"})
            neighbors = {item.path for item in self._dependency(dep_gap)}
        symbols = sorted(symbols, key=lambda item: (item.relative_path not in neighbors,
                                                    not item.exported, item.relative_path,
                                                    item.start_line, item.symbol_id))
        return [Candidate(path=s.relative_path, line=s.start_line, end_line=s.end_line,
                          channel="symbol") for s in symbols]

    def _lexical(self, query: str, limit: int):
        return self.exact.search(self.index, query, max_result=limit)

    @staticmethod
    def _exact_candidates(response) -> list[Candidate]:
        return [Candidate(path=item.relative_path,
                          line=next((m.line_number for m in item.matches if m.line_number is not None), 1),
                          channel="exact") for item in response.results]

    async def _hybrid(self, query: str, lexical, limit: int) -> list[Candidate]:
        if self.semantic is None:
            if "hybrid_lexical_only" not in self.diagnostics:
                self.diagnostics.append("hybrid_lexical_only")
            return self._exact_candidates(lexical)
        semantic = await self.semantic.search(query, top_k=limit)
        # Never trust embedding metadata as source evidence. Only paths/ranges
        # from the pinned catalog are admitted and subsequently read again.
        semantic = SemanticSearchResponse(query=query, results=[
            item for item in semantic.results if item.chunk.relative_path in self.catalog.files])
        fused = self.hybrid.fuse(query=query, lexical=lexical, semantic=semantic)
        exact_by_path = {c.path: c for c in self._exact_candidates(lexical)}
        result = []
        for item in fused.candidates:
            if item.best_semantic_chunk is not None:
                chunk = item.best_semantic_chunk
                result.append(Candidate(path=item.relative_path, line=chunk.start_line,
                                        end_line=chunk.end_line, channel="hybrid"))
            else:
                result.append(exact_by_path[item.relative_path].model_copy(update={"channel": "hybrid"}))
        return result

    async def find(self, gap: EvidenceGap) -> list[Candidate]:
        strategy = gap.strategy
        query = gap.query
        limit = min(60, self.budget.max_candidates_per_call * 4)
        candidates: list[Candidate] = []
        if strategy in {"path", "auto"} and query in self.catalog.files:
            candidates = [Candidate(path=query, channel="path")]
        elif strategy == "path":
            return []
        elif strategy == "dependency" or (strategy == "auto" and gap.source_path
                                           and (query.startswith(("./", "../", "@/")) or query == "*")):
            candidates = self._dependency(gap)
        elif strategy == "symbol" or (strategy == "auto" and re.fullmatch(r"[A-Za-z_$][\w$]*", query)):
            candidates = self._symbols(gap)
            if not candidates:
                candidates = self._exact_candidates(self._lexical(query, limit))
        elif strategy == "exact":
            candidates = self._exact_candidates(self._lexical(query, limit))
        elif strategy == "hybrid":
            candidates = await self._hybrid(query, self._lexical(query, limit), limit)
        else:
            lexical = self._lexical(query, limit)
            candidates = self._exact_candidates(lexical)
            if not candidates and self.semantic is not None:
                candidates = await self._hybrid(query, lexical, limit)
        # Stable deduplication and range checking; all output is a locator, not
        # source text. A malformed backend result cannot enter the read scope.
        unique: dict[tuple[str, int], Candidate] = {}
        for item in candidates:
            if item.path not in self.catalog.files:
                continue
            count = len(self.catalog.files[item.path].splitlines())
            if not 1 <= item.line <= count:
                continue
            end_line = min(item.end_line or (item.line + self.budget.max_window_lines - 1), count,
                           item.line + self.budget.max_window_lines - 1)
            if end_line < item.line:
                continue
            normalized = item.model_copy(update={"end_line": end_line})
            unique.setdefault((item.path, item.line), normalized)
        return list(unique.values())[:limit]
