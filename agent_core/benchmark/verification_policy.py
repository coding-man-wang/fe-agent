from __future__ import annotations

import shutil

from agent_core.coding.verification_command_model import AllowedVerificationCommand
from agent_core.coding.verification_policy import VerificationCommandPolicy


class BenchmarkVerificationPolicyError(RuntimeError):
    pass


def _which(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise BenchmarkVerificationPolicyError(f"找不到可执行文件: {name}")
    return resolved


def build_frontend_verification_policy() -> VerificationCommandPolicy:
    npm = _which("npm")
    commands = [
        AllowedVerificationCommand(
            command_id="frontend-lint",
            name="frontend lint",
            argv=[npm, "run", "lint"],
            timeout_seconds=180,
        ),
        AllowedVerificationCommand(
            command_id="frontend-typecheck",
            name="frontend typecheck",
            argv=[npm, "run", "typecheck"],
            timeout_seconds=180,
        ),
        AllowedVerificationCommand(
            command_id="frontend-unit-test",
            name="frontend unit test",
            argv=[npm, "run", "test"],
            timeout_seconds=240,
        ),
        AllowedVerificationCommand(
            command_id="frontend-build",
            name="frontend build",
            argv=[npm, "run", "build"],
            timeout_seconds=240,
        ),
        AllowedVerificationCommand(
            command_id="frontend-e2e",
            name="frontend e2e",
            argv=[npm, "run", "test:e2e"],
            timeout_seconds=600,
            max_output_chars=50_000,
        ),
    ]
    return VerificationCommandPolicy(commands)
