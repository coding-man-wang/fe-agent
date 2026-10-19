import re
from pathlib import Path

from agent_core.coding.handoff import (
    write_json_atomically,
)
from agent_core.git.combined_verification_model import (
    CombinedVerificationResult,
)


class CombinedVerificationStoreError(ValueError):
    pass


class CombinedVerificationStore:

    def __init__(
        self,
        repository_root: str | Path,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        self.directory = self.root / ".fe-agent" / "runtime" / "combined_verification"

    def _path(
        self,
        integration_id: str,
    ) -> Path:

        if not re.fullmatch(
            r"[0-9a-f]{64}",
            integration_id,
        ):

            raise (CombinedVerificationStoreError("非法 integration_id"))

        return self.directory / f"{integration_id}.json"

    def save(
        self,
        result: CombinedVerificationResult,
    ) -> None:

        path = self._path(result.integration_id)

        write_json_atomically(
            path,
            result.model_dump_json(indent=2),
        )

    def load(
        self,
        integration_id: str,
    ) -> CombinedVerificationResult:

        path = self._path(integration_id)

        if not path.is_file():

            raise (
                CombinedVerificationStoreError("Combined Verification " "结果不存在")
            )

        try:

            result = CombinedVerificationResult.model_validate_json(
                path.read_text(encoding="utf-8")
            )

        except Exception as exc:

            raise (
                CombinedVerificationStoreError("Combined Verification " "结果非法")
            ) from exc

        if result.integration_id != integration_id:

            raise (CombinedVerificationStoreError("Integration ID 不一致"))

        return result
