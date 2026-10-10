import hashlib
import subprocess
from pathlib import Path

import pytest

from agent_core.git.errors import GitWorkspaceConflictError
from agent_core.git.workspace import (
    AgentGitWorkspaceManager,
)
from agent_core.git.workspace_binding import (
    WorkspaceBindingService,
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
        "export const user = 1\n",
        encoding="utf-8",
    )

    run_git(
        root,
        "add",
        ".",
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


def create_proposal_id() -> str:
    return hashlib.sha256(b"proposal").hexdigest()


@pytest.mark.asyncio
async def test_bind_ready_workspace(
    tmp_path: Path,
):
    repository = tmp_path / "repo"

    repository.mkdir()

    base_sha = create_repository(repository)

    manager = AgentGitWorkspaceManager(
        repository_root=repository,
        workspace_root=(tmp_path / "worktrees"),
    )

    proposal_id = create_proposal_id()

    workspace = await manager.create(
        task_id="TASK-001",
        proposal_id=proposal_id,
        base_commit_sha=base_sha,
    )

    binding_service = WorkspaceBindingService()

    context = await binding_service.bind(
        workspace=workspace,
        task_id="TASK-001",
        proposal_id=proposal_id,
    )

    assert context.workspace_id == workspace.workspace_id

    assert context.execution_root == str(Path(workspace.worktree_path).resolve())

    assert context.source_repository_root == str(repository.resolve())


@pytest.mark.asyncio
async def test_bind_rejects_wrong_task(
    tmp_path: Path,
):
    repository = tmp_path / "repo"

    repository.mkdir()

    base_sha = create_repository(repository)

    proposal_id = create_proposal_id()

    manager = AgentGitWorkspaceManager(
        repository_root=repository,
        workspace_root=(tmp_path / "worktrees"),
    )

    workspace = await manager.create(
        task_id="TASK-001",
        proposal_id=proposal_id,
        base_commit_sha=base_sha,
    )

    service = WorkspaceBindingService()

    with pytest.raises(GitWorkspaceConflictError):
        await service.bind(
            workspace=workspace,
            task_id="TASK-999",
            proposal_id=proposal_id,
        )


@pytest.mark.asyncio
async def test_bind_rejects_wrong_proposal(
    tmp_path: Path,
):
    repository = tmp_path / "repo"

    repository.mkdir()

    base_sha = create_repository(repository)

    proposal_a = hashlib.sha256(b"proposal-a").hexdigest()

    proposal_b = hashlib.sha256(b"proposal-b").hexdigest()

    manager = AgentGitWorkspaceManager(
        repository_root=repository,
        workspace_root=(tmp_path / "worktrees"),
    )

    workspace = await manager.create(
        task_id="TASK-001",
        proposal_id=proposal_a,
        base_commit_sha=base_sha,
    )

    service = WorkspaceBindingService()

    with pytest.raises(GitWorkspaceConflictError):
        await service.bind(
            workspace=workspace,
            task_id="TASK-001",
            proposal_id=proposal_b,
        )
