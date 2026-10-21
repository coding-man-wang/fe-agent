from __future__ import annotations

import re

from agent_core.git.workspace import AgentGitWorkspaceManager


class RunScopedGitWorkspaceManager(AgentGitWorkspaceManager):
    """Keep benchmark branches unique across repeated runs."""

    def __init__(self, *, run_id: str, **kwargs) -> None:
        self.run_id = run_id
        super().__init__(**kwargs)

    def build_branch_name(self, *, task_id: str, workspace_id: str) -> str:
        task = re.sub(r"[^a-z0-9._-]+", "-", task_id.strip().lower()).strip("-._")
        task = task or "task"
        return (
            f"fe-agent-bench/{task}-"
            f"{self.run_id[:8]}-{workspace_id[:8]}"
        )
