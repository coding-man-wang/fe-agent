import re
from pathlib import Path

from agent_core.coding.handoff import (
    write_json_atomically,
)
from agent_core.coding.verification_record_model import (
    VerificationRunRecord,
)


class VerificationStoreError(ValueError):
    pass


class VerificationStore:
    """
    Verification Workflow 持久化。

    存储结构：

    .fe-agent/
    └── runtime/
        └── verifications/
            └── <proposal_id>.json
    """

    def __init__(
        self,
        repository_root: str | Path,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise VerificationStoreError("repository_root 不是目录")

        self.directory = self.root / ".fe-agent" / "runtime" / "verifications"

    def _path(
        self,
        proposal_id: str,
    ) -> Path:

        if not re.fullmatch(
            r"[0-9a-f]{64}",
            proposal_id,
        ):
            raise VerificationStoreError("非法 Proposal ID")

        return self.directory / f"{proposal_id}.json"

    def exists(
        self,
        proposal_id: str,
    ) -> bool:

        return self._path(proposal_id).exists()

    def save(
        self,
        record: VerificationRunRecord,
    ) -> None:

        path = self._path(record.proposal_id)

        write_json_atomically(
            path,
            record.model_dump_json(indent=2),
        )

    def load(
        self,
        proposal_id: str,
    ) -> VerificationRunRecord:

        path = self._path(proposal_id)

        if not path.is_file():
            raise VerificationStoreError("Verification Record 不存在")

        try:

            record = VerificationRunRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )

        except (
            OSError,
            ValueError,
        ) as exc:

            raise VerificationStoreError(
                "Verification Record " "无法读取或格式非法"
            ) from exc

        if record.proposal_id != proposal_id:
            raise VerificationStoreError("Verification Record " "的 Proposal ID 不一致")

        return record
