from __future__ import annotations

import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, Field

from agent_core.codebase.model import HybridRelevantFileSelection
from agent_core.codebase.reranker import Reranker, RerankDocument


class RerankFileScore(BaseModel):
    relative_path: str
    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)
    context_role: str
    has_symbol_evidence: bool
    source_truncated: bool


class TaskRerankDiagnosticResult(BaseModel):
    query: str
    status: Literal["complete", "incomplete", "error"]
    candidate_paths: list[str] = Field(default_factory=list)
    ranked_files: list[RerankFileScore] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class TaskRerankDiagnosticRunner:
    """
    只观察排名，不调用 AdaptiveRerankSelectionPolicy。
    不修改 Selection，也不使用 Benchmark Oracle。
    """

    def __init__(
        self,
        *,
        reranker: Reranker,
        max_document_chars: int = 6000,
    ):
        if max_document_chars < 1:
            raise ValueError("max_document_chars 必须 >= 1")

        self.reranker = reranker
        self.max_document_chars = max_document_chars

    def _build_document(
        self,
        *,
        root: Path,
        relative_path: str,
        category: str,
    ) -> tuple[RerankDocument, bool]:
        normalized = relative_path.replace("\\", "/")
        pure_path = PurePosixPath(normalized)

        if (
            pure_path.is_absolute()
            or PureWindowsPath(relative_path).drive
            or ".." in pure_path.parts
            or not pure_path.parts
        ):
            raise ValueError("非法 Repository Path")

        source_path = (root / Path(*pure_path.parts)).resolve(strict=True)

        if not source_path.is_relative_to(root) or not source_path.is_file():
            raise ValueError("文件超出 Repository 或不是文件")

        # 使用当前源码，不能盲信可能过期的 Semantic Chunk
        content = source_path.read_text(encoding="utf-8")
        truncated = len(content) > self.max_document_chars

        document = RerankDocument(
            document_id=relative_path,
            text=(
                f"path: {relative_path}\n"
                f"category: {category}\n"
                f"content:\n"
                f"{content[:self.max_document_chars]}"
            ),
        )

        return document, truncated

    async def run(
        self,
        *,
        query: str,
        repository_root: str | Path,
        selection: HybridRelevantFileSelection,
    ) -> TaskRerankDiagnosticResult:

        paths = [item.relative_path for item in selection.selected_files]

        def failed(status: str, issue: str):
            return TaskRerankDiagnosticResult(
                query=query,
                status=status,
                candidate_paths=paths,
                issues=[issue],
            )

        if not query.strip():
            return failed("error", "empty_query")

        if not paths or len(paths) != len(set(paths)):
            return failed("error", "invalid_candidate_paths")

        try:
            root = Path(repository_root).resolve(strict=True)
            if not root.is_dir():
                raise ValueError("Repository Root 不是目录")

            documents = []
            truncation = {}

            for item in selection.selected_files:
                document, truncated = self._build_document(
                    root=root,
                    relative_path=item.relative_path,
                    category=item.category,
                )
                documents.append(document)
                truncation[item.relative_path] = truncated

        except (OSError, UnicodeError, ValueError) as exc:
            return failed(
                "error",
                f"source_read_error:{type(exc).__name__}",
            )

        try:
            hits = await self.reranker.rerank(
                query=query,
                documents=documents,
                top_n=len(documents),
            )
        except Exception as exc:
            return failed(
                "error",
                f"reranker_error:{type(exc).__name__}",
            )

        expected = set(paths)
        returned_paths = [hit.document_id for hit in hits]
        returned_ranks = [hit.rank for hit in hits]

        issues = []

        if len(hits) != len(paths):
            issues.append("incomplete_rerank_response")

        if set(returned_paths) != expected:
            issues.append("candidate_set_mismatch")

        if len(returned_paths) != len(set(returned_paths)):
            issues.append("duplicate_document_id")

        if sorted(returned_ranks) != list(range(1, len(paths) + 1)):
            issues.append("invalid_ranks")

        if any(not math.isfinite(hit.score) for hit in hits):
            issues.append("non_finite_score")

        if issues:
            return TaskRerankDiagnosticResult(
                query=query,
                status="incomplete",
                candidate_paths=paths,
                issues=issues,
            )

        file_map = {item.relative_path: item for item in selection.selected_files}

        ranked_files = []

        for hit in sorted(hits, key=lambda x: x.rank):
            item = file_map[hit.document_id]

            ranked_files.append(
                RerankFileScore(
                    relative_path=hit.document_id,
                    rank=hit.rank,
                    score=hit.score,
                    context_role=item.context_role,
                    has_symbol_evidence=bool(item.symbol_candidates),
                    source_truncated=truncation[hit.document_id],
                )
            )

        return TaskRerankDiagnosticResult(
            query=query,
            status="complete",
            candidate_paths=paths,
            ranked_files=ranked_files,
        )
