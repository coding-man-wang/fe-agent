from pathlib import Path

from agent_core.benchmark.retrieval_model import (
    BenchmarkRetrievalCapture,
    BenchmarkRetrievedFile,
    BenchmarkTaskRetrieval,
)
from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.task_retrieval import (
    TaskRetrievalService,
)
from agent_core.sdd.model import (
    TaskPlan,
)


class BenchmarkRetrievalCaptureError(ValueError):
    pass


class BenchmarkRetrievalCaptureService:
    """
    Benchmark 专用 Retrieval Capture。

    职责：

        Repository
            ↓
        File Index
            ↓
        每个 Task 执行 Retrieval
            ↓
        转换为 Benchmark Retrieval Artifact

    本类不负责评价 Retrieval 是否正确。

    Recall / Precision 属于后续 Evaluator。
    """

    def __init__(
        self,
        *,
        repository_root: str | Path,
        file_indexer: RepositoryFileIndexer,
        retrieval_service: TaskRetrievalService,
    ) -> None:

        self.repository_root = Path(repository_root).resolve(strict=True)

        if not self.repository_root.is_dir():
            raise BenchmarkRetrievalCaptureError("repository_root 不是目录")

        self.file_indexer = file_indexer

        self.retrieval_service = retrieval_service

    async def capture(
        self,
        *,
        task_plan: TaskPlan,
    ) -> BenchmarkRetrievalCapture:

        # -----------------------------------
        # 1. 对真实 Benchmark Repository
        #    建立 File Index
        # -----------------------------------

        index = self.file_indexer.build(self.repository_root)

        captures: list[BenchmarkTaskRetrieval] = []

        # -----------------------------------
        # 2. 每个 Task 独立 Retrieval
        # -----------------------------------

        for task in task_plan.tasks:

            # Query Builder 本身也是
            # Benchmark 要记录的重要证据。
            queries = self.retrieval_service.build_queries(task)

            selection = await self.retrieval_service.retrieve(
                task=task,
                index=index,
            )

            selected_files = [
                BenchmarkRetrievedFile(
                    relative_path=(item.relative_path),
                    category=(item.category),
                    selection_score=(item.selection_score),
                    matched_queries=(list(item.matched_queries)),
                    reasons=(list(item.reasons)),
                )
                for item in selection.selected_files
            ]

            captures.append(
                BenchmarkTaskRetrieval(
                    task_id=(task.task_id),
                    task_type=(task.task_type),
                    title=(task.title),
                    queries=queries,
                    selected_files=(selected_files),
                    candidate_count=(selection.candidate_count),
                    selected_count=(selection.selected_count),
                )
            )

        # -----------------------------------
        # 3. 返回 Benchmark Artifact
        # -----------------------------------

        return BenchmarkRetrievalCapture(
            mode="lexical_baseline",
            indexed_files=(index.indexable_files),
            task_retrievals=(captures),
        )
