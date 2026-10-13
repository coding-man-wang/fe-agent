import subprocess
from pathlib import Path

import pytest

from agent_core.git.analysis_snapshot import (
    AnalysisSnapshotError,
    AnalysisSnapshotManager,
)
from agent_core.git.revision_code_snapshot import (
    RevisionCodeSnapshotFactory,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


def run_git(
    cwd: Path,
    *args: str,
) -> str:

    result = subprocess.run(
        [
            "git",
            *args,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def create_repository(
    root: Path,
) -> tuple[
    Path,
    str,
]:

    repository = root / "repository"

    repository.mkdir()

    run_git(
        repository,
        "init",
    )

    run_git(
        repository,
        "config",
        "user.name",
        "Test User",
    )

    run_git(
        repository,
        "config",
        "user.email",
        "test@example.com",
    )

    (repository / "package.json").write_text(
        """
{
  "name": "snapshot-test",
  "devDependencies": {
    "typescript": "5.0.0",
    "vite": "5.0.0"
  }
}
""".strip(),
        encoding="utf-8",
    )

    (repository / "src").mkdir()

    (repository / "src" / "user.ts").write_text(
        "export const version = 1;\n",
        encoding="utf-8",
    )

    run_git(
        repository,
        "add",
        ".",
    )

    run_git(
        repository,
        "commit",
        "-m",
        "initial",
    )

    sha = run_git(
        repository,
        "rev-parse",
        "HEAD",
    )

    return (
        repository,
        sha,
    )


def create_task() -> TaskDefinition:

    return TaskDefinition(
        task_id=("TASK-001-REV-A"),
        task_type="code_change",
        title="处理 Review 反馈",
        objective="修改用户逻辑",
        source_step_order=1,
        depends_on=[],
        inputs=["Review Feedback"],
        target_modules=["user"],
        expected_outputs=["Review Feedback 已处理"],
        verification=["测试通过"],
        acceptance_criteria_refs=["用户功能正常"],
    )


@pytest.mark.asyncio
async def test_create_detached_analysis_snapshot(
    tmp_path: Path,
):

    repository, sha = create_repository(tmp_path)

    snapshot_root = tmp_path / "snapshots"

    manager = AnalysisSnapshotManager(
        repository_root=repository,
        snapshot_root=(snapshot_root),
    )

    snapshot = await manager.create(
        task_id=("TASK-001-REV-A"),
        base_commit_sha=sha,
    )

    assert snapshot.status == "ready"

    snapshot_path = Path(snapshot.snapshot_path)

    assert snapshot_path.is_dir()

    actual_sha = run_git(
        snapshot_path,
        "rev-parse",
        "HEAD",
    )

    assert actual_sha == sha

    branch_result = subprocess.run(
        [
            "git",
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
        ],
        cwd=snapshot_path,
        capture_output=True,
        text=True,
    )

    assert branch_result.returncode == 1


@pytest.mark.asyncio
async def test_snapshot_is_pinned_to_old_commit(
    tmp_path: Path,
):

    repository, first_sha = create_repository(tmp_path)

    (repository / "src" / "user.ts").write_text(
        "export const version = 2;\n",
        encoding="utf-8",
    )

    run_git(
        repository,
        "add",
        ".",
    )

    run_git(
        repository,
        "commit",
        "-m",
        "version 2",
    )

    second_sha = run_git(
        repository,
        "rev-parse",
        "HEAD",
    )

    assert first_sha != second_sha

    manager = AnalysisSnapshotManager(
        repository_root=repository,
        snapshot_root=(tmp_path / "snapshots"),
    )

    snapshot = await manager.create(
        task_id=("TASK-001-REV-A"),
        base_commit_sha=(first_sha),
    )

    content = (Path(snapshot.snapshot_path) / "src" / "user.ts").read_text(
        encoding="utf-8"
    )

    assert content == "export const version = 1;\n"


@pytest.mark.asyncio
async def test_snapshot_reuse_is_deterministic(
    tmp_path: Path,
):

    repository, sha = create_repository(tmp_path)

    manager = AnalysisSnapshotManager(
        repository_root=repository,
        snapshot_root=(tmp_path / "snapshots"),
    )

    first = await manager.create(
        task_id=("TASK-001-REV-A"),
        base_commit_sha=sha,
    )

    second = await manager.create(
        task_id=("TASK-001-REV-A"),
        base_commit_sha=sha,
    )

    assert first.snapshot_id == second.snapshot_id

    assert first.snapshot_path == second.snapshot_path


@pytest.mark.asyncio
async def test_snapshot_rejects_mutated_worktree(
    tmp_path: Path,
):

    repository, sha = create_repository(tmp_path)

    manager = AnalysisSnapshotManager(
        repository_root=repository,
        snapshot_root=(tmp_path / "snapshots"),
    )

    snapshot = await manager.create(
        task_id=("TASK-001-REV-A"),
        base_commit_sha=sha,
    )

    (Path(snapshot.snapshot_path) / "src" / "user.ts").write_text(
        "mutated\n",
        encoding="utf-8",
    )

    with pytest.raises(
        AnalysisSnapshotError,
        match="已跟踪文件修改",
    ):
        await manager.create(
            task_id=("TASK-001-REV-A"),
            base_commit_sha=sha,
        )


@pytest.mark.asyncio
async def test_snapshot_rejects_unknown_commit(
    tmp_path: Path,
):

    repository, _ = create_repository(tmp_path)

    manager = AnalysisSnapshotManager(
        repository_root=repository,
        snapshot_root=(tmp_path / "snapshots"),
    )

    with pytest.raises(
        AnalysisSnapshotError,
        match="Base Commit 不存在",
    ):
        await manager.create(
            task_id=("TASK-001-REV-A"),
            base_commit_sha=("9" * 40),
        )


@pytest.mark.asyncio
async def test_revision_code_snapshot_uses_snapshot_root(
    tmp_path: Path,
):

    repository, sha = create_repository(tmp_path)

    manager = AnalysisSnapshotManager(
        repository_root=repository,
        snapshot_root=(tmp_path / "snapshots"),
    )

    factory = RevisionCodeSnapshotFactory(snapshot_manager=manager)

    result = await factory.create(
        task=create_task(),
        base_commit_sha=sha,
    )

    snapshot_root = Path(result.snapshot.snapshot_path).resolve()

    assert Path(result.repository.root_path).resolve() == snapshot_root

    assert Path(result.index.root_path).resolve() == snapshot_root

    indexed_paths = {item.relative_path for item in result.index.files}

    assert "src/user.ts" in indexed_paths

    assert "package.json" in indexed_paths
