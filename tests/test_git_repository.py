import subprocess
from pathlib import Path

import pytest

from agent_core.git.errors import (
    NotGitRepositoryError,
)
from agent_core.git.repository import (
    GitRepositoryInspector,
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
) -> None:
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
        "# Test\n",
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


@pytest.mark.asyncio
async def test_inspect_clean_repository(
    tmp_path: Path,
):
    create_git_repository(tmp_path)

    inspector = GitRepositoryInspector()

    state = await inspector.inspect(tmp_path)

    expected_head = run_git(
        tmp_path,
        "rev-parse",
        "HEAD",
    )

    assert state.repository_root == str(tmp_path.resolve())

    assert state.head_sha == expected_head

    assert state.is_detached_head is False

    assert state.branch_name is not None

    assert state.working_tree.is_clean is True


@pytest.mark.asyncio
async def test_inspect_dirty_tracked_file(
    tmp_path: Path,
):
    create_git_repository(tmp_path)

    (tmp_path / "README.md").write_text(
        "# Changed\n",
        encoding="utf-8",
    )

    state = await GitRepositoryInspector().inspect(tmp_path)

    assert state.working_tree.is_clean is False

    assert any("README.md" in line for line in state.working_tree.tracked_changes)


@pytest.mark.asyncio
async def test_inspect_untracked_file(
    tmp_path: Path,
):
    create_git_repository(tmp_path)

    (tmp_path / "new.txt").write_text(
        "new",
        encoding="utf-8",
    )

    state = await GitRepositoryInspector().inspect(tmp_path)

    assert state.working_tree.is_clean is False

    assert "new.txt" in state.working_tree.untracked_files


@pytest.mark.asyncio
async def test_inspect_from_nested_directory(
    tmp_path: Path,
):
    create_git_repository(tmp_path)

    nested = tmp_path / "src" / "features"

    nested.mkdir(parents=True)

    state = await GitRepositoryInspector().inspect(nested)

    assert state.repository_root == str(tmp_path.resolve())


@pytest.mark.asyncio
async def test_non_git_directory_is_rejected(
    tmp_path: Path,
):
    inspector = GitRepositoryInspector()

    with pytest.raises(NotGitRepositoryError):
        await inspector.inspect(tmp_path)


@pytest.mark.asyncio
async def test_detached_head_is_supported(
    tmp_path: Path,
):
    create_git_repository(tmp_path)

    sha = run_git(
        tmp_path,
        "rev-parse",
        "HEAD",
    )

    run_git(
        tmp_path,
        "checkout",
        "--detach",
        sha,
    )

    state = await GitRepositoryInspector().inspect(tmp_path)

    print(state)
    assert state.head_sha == sha

    assert state.is_detached_head is True

    assert state.branch_name is None
