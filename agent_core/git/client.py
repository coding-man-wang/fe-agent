from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class GitCommandResult:
    exit_code: int
    stdout: str
    stderr: str


class GitClient(Protocol):
    """
    Git 基础执行能力。

    这一层只描述：
    “如何执行一个明确的 Git argv”。

    不包含：
    - Repository Policy
    - Workspace Policy
    - Branch 命名规则
    - Task / Proposal 业务语义
    """

    async def run(
        self,
        *,
        cwd: Path,
        args: list[str],
        allow_failure: bool = False,
    ) -> GitCommandResult: ...
