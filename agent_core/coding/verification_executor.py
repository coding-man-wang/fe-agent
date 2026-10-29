import asyncio
import os
from pathlib import Path

from agent_core.coding.verification_command_model import (
    AllowedVerificationCommand,
    VerificationCommandResult,
)
from agent_core.coding.verification_model import (
    PatchVerificationResult,
    VerificationCheck,
)
from agent_core.coding.verification_plan import (
    VerificationPlan,
)
from agent_core.coding.verification_policy import (
    VerificationCommandPolicy,
)


class VerificationExecutionError(ValueError):
    pass


class VerificationExecutor:

    def __init__(
        self,
        *,
        repository_root: str | Path,
        policy: VerificationCommandPolicy,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise VerificationExecutionError("repository_root 不是目录")

        self.policy = policy

    def _resolve_working_directory(
        self,
        relative_path: str,
    ) -> Path:

        if (
            not relative_path
            or "\\" in relative_path
            or ":" in relative_path
            or "\x00" in relative_path
        ):
            raise VerificationExecutionError("非法 working_directory")

        candidate = (self.root / relative_path).resolve(strict=True)

        if not candidate.is_relative_to(self.root):
            raise VerificationExecutionError("working_directory 位于仓库外")

        if not candidate.is_dir():
            raise VerificationExecutionError("working_directory 不是目录")

        return candidate

    @staticmethod
    def _build_environment() -> dict[
        str,
        str,
    ]:
        """
        不把完整进程环境无条件传给子进程。

        这里只保留运行 CLI 常用的基础变量。
        """

        allowed_names = {
            "PATH",
            "PATHEXT",
            "SystemRoot",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "HOME",
            "USERPROFILE",
            "LANG",
            "LC_ALL",
            "CI",
        }

        return {key: value for key, value in os.environ.items() if key in allowed_names}

    @staticmethod
    def _truncate(
        text: str,
        *,
        limit: int,
    ) -> tuple[str, bool]:

        if len(text) <= limit:
            return (
                text,
                False,
            )

        return (
            text[:limit] + "\n...[output truncated]",
            True,
        )

    async def _run_command(
        self,
        command: AllowedVerificationCommand,
    ) -> VerificationCommandResult:

        cwd = self._resolve_working_directory(command.working_directory)

        try:
            process = await asyncio.create_subprocess_exec(
                *command.argv,
                cwd=str(cwd),
                env=self._build_environment(),
                stdout=(asyncio.subprocess.PIPE),
                stderr=(asyncio.subprocess.PIPE),
            )

        except (
            OSError,
            ValueError,
        ) as exc:

            return VerificationCommandResult(
                command_id=(command.command_id),
                name=command.name,
                status="failed",
                exit_code=None,
                stderr=(f"{type(exc).__name__}: " f"{exc}"),
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=(command.timeout_seconds),
            )

        except asyncio.TimeoutError:

            process.kill()

            await process.communicate()

            return VerificationCommandResult(
                command_id=(command.command_id),
                name=command.name,
                status="timed_out",
                exit_code=None,
                stderr=("Verification command " "timed out"),
            )

        stdout = stdout_bytes.decode(
            "utf-8",
            errors="replace",
        )

        stderr = stderr_bytes.decode(
            "utf-8",
            errors="replace",
        )

        stdout, stdout_truncated = self._truncate(
            stdout,
            limit=(command.max_output_chars),
        )

        stderr, stderr_truncated = self._truncate(
            stderr,
            limit=(command.max_output_chars),
        )

        return VerificationCommandResult(
            command_id=(command.command_id),
            name=command.name,
            status=("passed" if process.returncode == 0 else "failed"),
            exit_code=(process.returncode),
            stdout=stdout,
            stderr=stderr,
            truncated=(stdout_truncated or stderr_truncated),
        )

    async def run(
        self,
        *,
        plan: VerificationPlan,
    ) -> PatchVerificationResult:

        if not plan.command_ids:
            raise VerificationExecutionError("Verification Plan 为空")

        # 先全部 resolve。
        # 如果有一个 command_id 未授权，
        # 任何命令都不要执行。
        commands = [self.policy.resolve(command_id) for command_id in plan.command_ids]

        results: list[VerificationCommandResult] = []

        for command in commands:

            result = await self._run_command(command)

            results.append(result)

            # ==================================
            # Verification Execution Strategy
            #
            # fail_fast:
            #   首个失败立即停止。
            #
            # collect_all:
            #   即使已有失败，
            #   仍继续收集后续 Verification
            #   的真实执行结果。
            #
            # Test Task 使用 collect_all，
            # 目的是让 Repair 一次看到：
            #
            # - structural failures
            # - behavioral failures
            #
            # 避免前置 lint failure
            # 隐藏后续 Unit / E2E failure。
            # ==================================

            if result.status != "passed" and plan.execution_mode == "fail_fast":
                break

        checks = [
            VerificationCheck(
                command_id=result.command_id,
                name=result.name,
                status=("passed" if result.status == "passed" else "failed"),
                details=self._format_details(result),
            )
            for result in results
        ]

        passed = len(results) == len(commands) and all(
            item.status == "passed" for item in results
        )

        return PatchVerificationResult(
            proposal_id=(plan.proposal_id),
            task_id=(plan.task_id),
            status=("passed" if passed else "failed"),
            checks=checks,
        )

    @staticmethod
    def _format_details(
        result: VerificationCommandResult,
    ) -> str:

        parts = [
            ("command_id=" f"{result.command_id}"),
            ("status=" f"{result.status}"),
        ]

        if result.exit_code is not None:
            parts.append("exit_code=" f"{result.exit_code}")

        if result.stdout:
            parts.append("stdout:\n" + result.stdout)

        if result.stderr:
            parts.append("stderr:\n" + result.stderr)

        if result.truncated:
            parts.append("output_truncated=true")

        return "\n".join(parts)
