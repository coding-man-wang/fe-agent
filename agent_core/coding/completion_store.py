import re
from pathlib import Path

from agent_core.coding.handoff import (
    write_json_atomically,
)
from agent_core.coding.task_completion_model import (
    TaskCompletionRecord,
)


class CompletionStoreError(ValueError):
    pass


class TaskCompletionStore:

    def __init__(
        self,
        repository_root: str | Path,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        self.directory = self.root / ".fe-agent" / "runtime" / "completions"

    def _path(
        self,
        proposal_id: str,
    ) -> Path:

        if not re.fullmatch(
            r"[0-9a-f]{64}",
            proposal_id,
        ):
            raise CompletionStoreError("非法 Proposal ID")

        return self.directory / f"{proposal_id}.json"

    def exists(
        self,
        proposal_id: str,
    ) -> bool:

        return self._path(proposal_id).exists()

    def save(
        self,
        record: TaskCompletionRecord,
    ) -> None:

        write_json_atomically(
            self._path(record.proposal_id),
            record.model_dump_json(indent=2),
        )

    def load(
        self,
        proposal_id: str,
    ) -> TaskCompletionRecord:

        path = self._path(proposal_id)

        if not path.is_file():
            raise CompletionStoreError("Task Completion Record 不存在")

        try:

            record = TaskCompletionRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )

        except (
            OSError,
            ValueError,
        ) as exc:

            raise CompletionStoreError("Task Completion Record " "格式非法") from exc

        if record.proposal_id != proposal_id:
            raise CompletionStoreError("Completion Proposal ID 不一致")

        return record
