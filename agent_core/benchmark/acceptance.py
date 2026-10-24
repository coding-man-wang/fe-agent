from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from agent_core.coding.verification_command_model import AllowedVerificationCommand
from agent_core.coding.verification_executor import VerificationExecutor
from agent_core.coding.verification_model import PatchVerificationResult
from agent_core.coding.verification_plan import VerificationPlan
from agent_core.coding.verification_policy import VerificationCommandPolicy


class HiddenAcceptanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class HiddenAcceptanceSpec:
    fixture_name: str
    relative_path: str


class HiddenAcceptanceRegistry:

    SPECS: dict[
        str,
        HiddenAcceptanceSpec,
    ] = {
        "CASE-001": (
            HiddenAcceptanceSpec(
                fixture_name=("CASE-001.test.ts"),
                relative_path=(
                    "tests/unit/" "__fe_agent_hidden__/" "CASE-001.acceptance.test.ts"
                ),
            )
        ),
        "CASE-006": (
            HiddenAcceptanceSpec(
                fixture_name=("CASE-006.test.ts"),
                relative_path=(
                    "tests/unit/" "__fe_agent_hidden__/" "CASE-006.acceptance.test.ts"
                ),
            )
        ),
    }

    def get(
        self,
        case_id: str,
    ) -> HiddenAcceptanceSpec:

        spec = self.SPECS.get(case_id)

        if spec is None:
            raise HiddenAcceptanceError(
                "v1.1 尚未为 " f"{case_id} " "定义 hidden acceptance test"
            )

        return spec


class HiddenAcceptanceRunner:
    def __init__(
        self,
        *,
        fixture_root: str | Path,
        registry: HiddenAcceptanceRegistry | None = None,
    ) -> None:
        self.fixture_root = Path(fixture_root).resolve(strict=True)
        self.registry = registry or HiddenAcceptanceRegistry()

    async def run(
        self,
        *,
        case_id: str,
        repository_root: str | Path,
    ) -> PatchVerificationResult:
        root = Path(repository_root).resolve(strict=True)
        spec = self.registry.get(case_id)
        fixture = self.fixture_root / spec.fixture_name
        if not fixture.is_file():
            raise HiddenAcceptanceError(f"Hidden acceptance fixture 不存在: {fixture}")

        test_path = root / spec.relative_path
        test_path.parent.mkdir(parents=True, exist_ok=True)
        if test_path.exists():
            raise HiddenAcceptanceError(
                f"Hidden test path 已存在: {spec.relative_path}"
            )

        npm = shutil.which("npm")
        if not npm:
            raise HiddenAcceptanceError("找不到 npm")

        proposal_id = hashlib.sha256(
            f"{case_id}:hidden-acceptance-v1".encode("utf-8")
        ).hexdigest()
        policy = VerificationCommandPolicy(
            [
                AllowedVerificationCommand(
                    command_id="benchmark-hidden-acceptance",
                    name=f"{case_id} hidden acceptance",
                    argv=[
                        npm,
                        "exec",
                        "--",
                        "vitest",
                        "run",
                        spec.relative_path,
                    ],
                    timeout_seconds=240,
                    max_output_chars=50_000,
                )
            ]
        )
        executor = VerificationExecutor(repository_root=root, policy=policy)

        try:
            test_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
            return await executor.run(
                plan=VerificationPlan(
                    proposal_id=proposal_id,
                    task_id=case_id,
                    command_ids=["benchmark-hidden-acceptance"],
                )
            )
        finally:
            if test_path.exists():
                test_path.unlink()
            current = test_path.parent
            if current.name == "__fe_agent_hidden__":
                try:
                    current.rmdir()
                except OSError:
                    pass
