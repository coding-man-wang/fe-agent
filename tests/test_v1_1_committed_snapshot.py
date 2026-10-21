from pathlib import Path
import shutil

import pytest

from agent_core.benchmark.committed_snapshot import CommittedSnapshotWorktreeManager
from agent_core.git.client import GitCommandResult


class FakeGitClient:
    def __init__(self, commit_sha: str) -> None:
        self.commit_sha = commit_sha
        self.calls: list[list[str]] = []

    async def run(self, *, cwd: Path, args: list[str], allow_failure: bool = False):
        self.calls.append(list(args))
        if args[:3] == ["worktree", "add", "--detach"]:
            target = Path(args[3])
            target.mkdir(parents=True)
            (target / "marker.txt").write_text("snapshot", encoding="utf-8")
            return GitCommandResult(0, "", "")
        if args == ["rev-parse", "HEAD"]:
            return GitCommandResult(0, self.commit_sha + "\n", "")
        if args[:3] == ["worktree", "remove", "--force"]:
            target = Path(args[3])
            if target.exists():
                shutil.rmtree(target)
            return GitCommandResult(0, "", "")
        raise AssertionError(f"unexpected git args: {args}")


@pytest.mark.asyncio
async def test_committed_snapshot_is_materialized_from_exact_commit(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    commit_sha = "a" * 40
    client = FakeGitClient(commit_sha)
    manager = CommittedSnapshotWorktreeManager(
        repository_root=repository,
        snapshot_root=tmp_path / "snapshots",
        git_client=client,
    )

    snapshot = await manager.create(task_id="TASK-002", commit_sha=commit_sha)

    assert snapshot.is_dir()
    assert (snapshot / "marker.txt").read_text(encoding="utf-8") == "snapshot"
    assert client.calls[0][0:3] == ["worktree", "add", "--detach"]
    assert client.calls[0][-1] == commit_sha

    await manager.remove(snapshot)
    assert not snapshot.exists()
