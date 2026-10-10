import hashlib
import subprocess
from pathlib import Path

import pytest

from agent_core.git.commit_service import (
    AgentGitCommitService,
)
from agent_core.git.workspace import (
    AgentGitWorkspaceManager,
)


def run_git(
    root: Path,
    *args: str,
) -> str:

    result = subprocess.run(
        [
            "git",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    return result.stdout.strip()


def create_repository(
    root: Path,
) -> str:

    run_git(
        root,
        "init",
    )

    run_git(
        root,
        "config",
        "user.email",
        "agent-test@example.com",
    )

    run_git(
        root,
        "config",
        "user.name",
        "Agent Test",
    )

    (root / "user.ts").write_text(
        "export const value = 1\n",
        encoding="utf-8",
    )

    run_git(
        root,
        "add",
        "--all",
    )

    run_git(
        root,
        "commit",
        "-m",
        "initial",
    )

    return run_git(
        root,
        "rev-parse",
        "HEAD",
    )


@pytest.mark.asyncio
async def test_commit_agent_workspace(
    tmp_path: Path,
):

    repository = tmp_path / "repo"

    repository.mkdir()

    base_sha = create_repository(repository)

    proposal_id = hashlib.sha256(b"proposal").hexdigest()

    manager = AgentGitWorkspaceManager(
        repository_root=repository,
        workspace_root=(tmp_path / "worktrees"),
    )

    workspace = await manager.create(
        task_id="TASK-001",
        proposal_id=proposal_id,
        base_commit_sha=base_sha,
    )

    worktree = Path(workspace.worktree_path)

    (worktree / "user.ts").write_text(
        "export const value = 2\n",
        encoding="utf-8",
    )

    service = AgentGitCommitService()

    result = await service.commit(
        workspace=workspace,
        message=("feat: update user value"),
    )

    assert result.parent_sha == base_sha

    assert result.commit_sha != base_sha

    assert (
        run_git(
            worktree,
            "rev-parse",
            "HEAD",
        )
        == result.commit_sha
    )

    assert (
        run_git(
            worktree,
            "status",
            "--porcelain",
        )
        == ""
    )
    assert (
        run_git(
            repository,
            "rev-parse",
            "HEAD",
        )
        == base_sha
    )

    assert (
        run_git(
            repository,
            "rev-parse",
            "HEAD",
        )
        == base_sha
    )


@pytest.mark.asyncio
async def test_commit_rejects_clean_workspace(
    tmp_path: Path,
):

    repository = tmp_path / "repo"

    repository.mkdir()

    base_sha = create_repository(repository)

    manager = AgentGitWorkspaceManager(
        repository_root=repository,
        workspace_root=(tmp_path / "worktrees"),
    )

    workspace = await manager.create(
        task_id="TASK-001",
        proposal_id=(hashlib.sha256(b"proposal").hexdigest()),
        base_commit_sha=base_sha,
    )

    service = AgentGitCommitService()

    with pytest.raises(Exception):
        await service.commit(
            workspace=workspace,
            message="feat: no change",
        )
