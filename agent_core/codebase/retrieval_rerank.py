from pathlib import Path

from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
)
from agent_core.codebase.reranker import (
    RerankDocument,
    Reranker,
)
from agent_core.codebase.rerank_selection import (
    AdaptiveRerankSelectionPolicy,
)


class RetrievalRerankService:

    def __init__(
        self,
        *,
        reranker: Reranker,
        selection_policy: AdaptiveRerankSelectionPolicy | None = None,
        max_document_chars: int = 6000,
    ) -> None:

        if max_document_chars < 1:

            raise ValueError("max_document_chars " "必须 >= 1")

        self.reranker = reranker

        self.selection_policy = selection_policy or AdaptiveRerankSelectionPolicy()

        self.max_document_chars = max_document_chars

    async def rerank(
        self,
        *,
        query: str,
        repository_root: str,
        selection: HybridRelevantFileSelection,
        top_n: int,
    ) -> HybridRelevantFileSelection:

        if not (selection.selected_files):
            return selection

        documents = [
            RerankDocument(
                document_id=(item.relative_path),
                text=(
                    self._build_document(
                        repository_root=(repository_root),
                        item=item,
                    )
                ),
            )
            for item in selection.selected_files
        ]

        hits = await self.reranker.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
        )
        hits = self.selection_policy.select(hits)
        file_map = {item.relative_path: item for item in selection.selected_files}

        ranked_files = []

        for hit in hits:

            item = file_map.get(hit.document_id)

            if item is None:
                continue

            ranked_files.append(
                item.model_copy(
                    deep=True,
                    update={
                        "rerank_score": (hit.score),
                        "rerank_rank": (hit.rank),
                    },
                )
            )

        selected_paths = {item.relative_path for item in ranked_files}

        skipped = list(selection.skipped_by_budget)

        skipped.extend(
            item.relative_path
            for item in selection.selected_files
            if item.relative_path not in selected_paths
        )

        return selection.model_copy(
            deep=True,
            update={
                "selected_count": (len(ranked_files)),
                "total_selected_bytes": sum(item.size_bytes for item in ranked_files),
                "selected_files": (ranked_files),
                "skipped_by_budget": (list(dict.fromkeys(skipped))),
            },
        )

    def _build_document(
        self,
        *,
        repository_root: str,
        item: HybridRelevantFile,
    ) -> str:

        content = ""

        if item.best_semantic_chunk is not None:

            content = item.best_semantic_chunk.content

        else:

            path = Path(repository_root) / item.relative_path

            try:

                content = path.read_text(encoding="utf-8")

            except (
                OSError,
                UnicodeError,
            ):

                content = ""

        content = content[: self.max_document_chars]

        return (
            f"path: "
            f"{item.relative_path}\n"
            f"category: "
            f"{item.category}\n"
            f"content:\n"
            f"{content}"
        )
