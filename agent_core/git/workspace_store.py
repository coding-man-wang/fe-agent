import os
import tempfile
from pathlib import Path

from agent_core.git.workspace_model import (
    GitWorkspaceRecord,
)


class GitWorkspaceStore:

    def __init__(
            self,
            repository_root: str | Path,
    ) -> None:

        self.repository_root = Path(repository_root).resolve()

        self.directory = (
                self.repository_root / ".fe-agent" / "runtime" / "git_workspaces"
        )

    def _path(
            self,
            workspace_id: str,
    ) -> Path:

        self._validate_id(workspace_id)

        return self.directory / f"{workspace_id}.json"

    def exists(
            self,
            workspace_id: str,
    ) -> bool:

        return self._path(workspace_id).exists()

    def save(
            self,
            record: GitWorkspaceRecord,
    ) -> None:

        path = self._path(record.workspace_id)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
            ) as file:

                temporary_path = Path(file.name)

                file.write(record.model_dump_json(indent=2))

                file.flush()

                os.fsync(file.fileno())

            os.replace(
                temporary_path,
                path,
            )

        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def load(
            self,
            workspace_id: str,
    ) -> GitWorkspaceRecord:

        path = self._path(workspace_id)

        return GitWorkspaceRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def _validate_id(
            self,
            workspace_id: str,
    ) -> None:

        if len(workspace_id) != 64 or any(
                char not in "0123456789abcdef" for char in workspace_id
        ):
            raise ValueError("非法 Workspace ID")
