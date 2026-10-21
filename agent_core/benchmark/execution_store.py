from __future__ import annotations

import os
import tempfile
from pathlib import Path

from agent_core.benchmark.execution_model import FullExecutionRunRecord


class FullExecutionRunStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.directory = self.root / ".fe-agent" / "benchmark" / "full_execution" / "runs"

    def save(self, record: FullExecutionRunRecord) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{record.run_id}.json"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=f".{record.run_id}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary = Path(file.name)
                file.write(record.model_dump_json(indent=2))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return path
