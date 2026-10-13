import os
from pathlib import Path

from agent_core.git.analysis_snapshot_model import (
    AnalysisSnapshotRecord,
)


class AnalysisSnapshotStoreError(ValueError):
    pass


class AnalysisSnapshotStore:

    def __init__(
        self,
        repository_root: str | Path,
    ) -> None:

        self.repository_root = Path(repository_root).resolve(strict=True)

        self.store_dir = (
            self.repository_root / ".fe-agent" / "runtime" / "analysis_snapshots"
        )

    def _validate_id(
        self,
        snapshot_id: str,
    ) -> None:

        if len(snapshot_id) != 64 or any(
            char not in "0123456789abcdef" for char in snapshot_id
        ):
            raise AnalysisSnapshotStoreError("snapshot_id 非法")

    def _path(
        self,
        snapshot_id: str,
    ) -> Path:

        self._validate_id(snapshot_id)

        return self.store_dir / f"{snapshot_id}.json"

    def exists(
        self,
        snapshot_id: str,
    ) -> bool:

        return self._path(snapshot_id).is_file()

    def save(
        self,
        record: AnalysisSnapshotRecord,
    ) -> None:

        path = self._path(record.snapshot_id)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = path.with_suffix(".tmp")

        content = record.model_dump_json(indent=2)

        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(content)

            file.flush()

            os.fsync(file.fileno())

        os.replace(
            temp_path,
            path,
        )

    def load(
        self,
        snapshot_id: str,
    ) -> AnalysisSnapshotRecord:

        path = self._path(snapshot_id)

        if not path.is_file():
            raise AnalysisSnapshotStoreError("Analysis Snapshot " "不存在")

        try:

            record = AnalysisSnapshotRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )

        except Exception as exc:

            raise AnalysisSnapshotStoreError("Analysis Snapshot " "记录无效") from exc

        if record.snapshot_id != snapshot_id:
            raise AnalysisSnapshotStoreError("Snapshot ID 不一致")

        return record
