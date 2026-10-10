import hashlib
import subprocess
from pathlib import Path

import pytest

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


def create_git_repository(
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

    (root / "README.md").write_text(
        "# committed\n",
        encoding="utf-8",
    )

    run_git(
        root,
        "add",
        "README.md",
    )

    run_git(
        root,
        "commit",
        "-m",
        "initial commit",
    )

    return run_git(
        root,
        "rev-parse",
        "HEAD",
    )


def proposal_id() -> str:
    return hashlib.sha256(b"proposal-001").hexdigest()


@pytest.mark.asyncio
async def test_create_agent_git_workspace(
    tmp_path: Path,
):
    repository = tmp_path / "repository"

    repository.mkdir()

    base_sha = create_git_repository(repository)

    workspace_root = tmp_path / "agent-worktrees"

    manager = AgentGitWorkspaceManager(
        repository_root=(repository),
        workspace_root=(workspace_root),
    )

    record = await manager.create(
        task_id="TASK-001",
        proposal_id=(proposal_id()),
        base_commit_sha=(base_sha),
    )

    assert record.status == "ready"

    assert record.base_commit_sha == base_sha

    worktree = Path(record.worktree_path)

    assert worktree.is_dir()

    assert (
        run_git(
            worktree,
            "rev-parse",
            "HEAD",
        )
        == base_sha
    )

    assert (
        run_git(
            worktree,
            "branch",
            "--show-current",
        )
        == record.branch_name
    )


@pytest.mark.asyncio
async def test_workspace_is_based_on_committed_head(
    tmp_path: Path,
):
    repository = tmp_path / "repository"

    repository.mkdir()

    base_sha = create_git_repository(repository)

    # Developer 当前有未提交修改。
    (repository / "README.md").write_text(
        "# developer local change\n",
        encoding="utf-8",
    )

    manager = AgentGitWorkspaceManager(
        repository_root=(repository),
        workspace_root=(tmp_path / "agent-worktrees"),
    )

    record = await manager.create(
        task_id="TASK-001",
        proposal_id=(proposal_id()),
        base_commit_sha=base_sha,
    )

    worktree = Path(record.worktree_path)

    content = (worktree / "README.md").read_text(encoding="utf-8")

    assert content == "# committed\n"

    assert (repository / "README.md").read_text(
        encoding="utf-8"
    ) == "# developer local change\n"


@pytest.mark.asyncio
async def test_create_workspace_is_idempotent(
    tmp_path: Path,
):
    repository = tmp_path / "repository"

    repository.mkdir()

    base_sha = create_git_repository(repository)

    manager = AgentGitWorkspaceManager(
        repository_root=repository,
        workspace_root=(tmp_path / "agent-worktrees"),
    )

    first = await manager.create(
        task_id="TASK-001",
        proposal_id=proposal_id(),
        base_commit_sha=base_sha,
    )

    second = await manager.create(
        task_id="TASK-001",
        proposal_id=proposal_id(),
        base_commit_sha=base_sha,
    )

    assert second.workspace_id == first.workspace_id

    assert second.worktree_path == first.worktree_path

    assert second.branch_name == first.branch_name
