from dataclasses import dataclass
from pathlib import Path

from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.model import (
    RepositoryFileIndex,
    RepositoryProfile,
)
from agent_core.codebase.scanner import (
    RepositoryScanner,
)
from agent_core.git.analysis_snapshot import (
    AnalysisSnapshotManager,
)
from agent_core.git.analysis_snapshot_model import (
    AnalysisSnapshotRecord,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


@dataclass(frozen=True)
class RevisionCodeSnapshot:

    snapshot: AnalysisSnapshotRecord

    repository: RepositoryProfile

    index: RepositoryFileIndex


class RevisionCodeSnapshotFactory:

    def __init__(
        self,
        *,
        snapshot_manager: AnalysisSnapshotManager,
        scanner: RepositoryScanner | None = None,
        indexer: RepositoryFileIndexer | None = None,
    ) -> None:

        self.snapshot_manager = snapshot_manager

        self.scanner = scanner or RepositoryScanner()

        self.indexer = indexer or RepositoryFileIndexer()

    async def create(
        self,
        *,
        task: TaskDefinition,
        base_commit_sha: str,
    ) -> RevisionCodeSnapshot:

        snapshot = await self.snapshot_manager.create(
            task_id=task.task_id,
            base_commit_sha=(base_commit_sha),
        )

        root = Path(snapshot.snapshot_path).resolve(strict=True)

        repository = self.scanner.scan(root)

        index = self.indexer.build(root)

        if Path(repository.root_path).resolve() != root:
            raise ValueError("RepositoryProfile " "没有基于 Revision Snapshot")

        if Path(index.root_path).resolve() != root:
            raise ValueError("RepositoryFileIndex " "没有基于 Revision Snapshot")

        return RevisionCodeSnapshot(
            snapshot=snapshot,
            repository=repository,
            index=index,
        )
