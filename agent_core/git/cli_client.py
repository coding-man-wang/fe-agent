import asyncio
from pathlib import Path

from agent_core.git.client import (
    GitCommandResult,
)
from agent_core.git.errors import (
    GitCommandError,
)


class CliGitClient:
    """
    基于系统 Git CLI 的 GitClient 实现。

    通过 create_subprocess_exec 执行，
    不经过 shell。
    """

    async def run(
            self,
            *,
            cwd: Path,
            args: list[str],
            allow_failure: bool = False,
    ) -> GitCommandResult:
        process = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_bytes, stderr_bytes = await process.communicate()

        stdout = stdout_bytes.decode(
            "utf-8",
            errors="replace",
        )

        stderr = stderr_bytes.decode(
            "utf-8",
            errors="replace",
        )

        result = GitCommandResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

        if result.exit_code != 0 and not allow_failure:
            raise GitCommandError(
                args=args,
                exit_code=result.exit_code,
                stderr=result.stderr,
            )

        return result
