import re
from pathlib import Path

from agent_core.coding.execution_journal_model import (
    PatchExecutionJournal,
)
from agent_core.coding.handoff import (
    write_json_atomically,
)


class ExecutionJournalError(ValueError):
    pass


class ExecutionJournalStore:

    def __init__(
        self,
        repository_root: str | Path,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        self.directory = self.root / ".fe-agent" / "runtime" / "executions"

    def _path(
        self,
        proposal_id: str,
    ) -> Path:

        if not re.fullmatch(
            r"[0-9a-f]{64}",
            proposal_id,
        ):
            raise ExecutionJournalError("非法 Proposal ID")

        return self.directory / f"{proposal_id}.json"

    def create(
        self,
        journal: PatchExecutionJournal,
    ) -> None:

        path = self._path(journal.proposal_id)

        if path.exists():
            raise ExecutionJournalError("该 Proposal 已存在 Execution Journal")

        write_json_atomically(
            path,
            journal.model_dump_json(indent=2),
        )

    def save(
        self,
        journal: PatchExecutionJournal,
    ) -> None:

        path = self._path(journal.proposal_id)

        if not path.exists():
            raise ExecutionJournalError("Execution Journal 不存在")

        write_json_atomically(
            path,
            journal.model_dump_json(indent=2),
        )

    def load(
        self,
        proposal_id: str,
    ) -> PatchExecutionJournal:

        path = self._path(proposal_id)

        if not path.exists():
            raise ExecutionJournalError("Execution Journal 不存在")

        return PatchExecutionJournal.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def exists(
        self,
        proposal_id: str,
    ) -> bool:

        return self._path(proposal_id).exists()
