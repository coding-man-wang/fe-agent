import hashlib
import subprocess
from pathlib import Path

import pytest

from agent_core.git.handoff_model import (
    GitHandoffFile,
    GitHandoffRecord,
)
from agent_core.git.push_service import (
    AgentGitPushService,
    GitPushError,
)

# ============================================================
# Git Test Helpers
# ============================================================


def run_git(
    root: Path,
    *args: str,
) -> str:
    """
    在普通 Git Repository 中执行 Git 命令。
    """

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


def run_bare_git(
    repository: Path,
    *args: str,
) -> str:
    """
    在 Bare Repository 中执行 Git 命令。

    Bare Repository 没有 Working Tree，
    所以通过 --git-dir 指定仓库。
    """

    result = subprocess.run(
        [
            "git",
            "--git-dir",
            str(repository),
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    return result.stdout.strip()


def create_repository(
    root: Path,
) -> str:
    """
    创建一个真实 Git Repository，
    并生成第一个 Commit。

    返回：
        initial commit sha
    """

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def create_bare_remote(
    root: Path,
) -> Path:
    """
    创建本地 Bare Repository，
    用来模拟 origin。

    不需要访问真正 GitHub，
    但仍然会执行真实 git push。
    """

    remote = root / "remote.git"

    subprocess.run(
        [
            "git",
            "init",
            "--bare",
            str(remote),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    return remote


def create_agent_commit(
    repository: Path,
    *,
    branch_name: str,
) -> tuple[str, str]:
    """
    在 source repository 中创建 Agent Branch，
    修改 user.ts 并提交。

    返回：
        base_commit_sha
        agent_commit_sha
    """

    base_sha = run_git(
        repository,
        "rev-parse",
        "HEAD",
    )

    run_git(
        repository,
        "checkout",
        "-b",
        branch_name,
    )

    (repository / "user.ts").write_text(
        "export const value = 2\n",
        encoding="utf-8",
    )

    run_git(
        repository,
        "add",
        "--all",
    )

    run_git(
        repository,
        "commit",
        "-m",
        "agent change",
    )

    agent_commit_sha = run_git(
        repository,
        "rev-parse",
        "HEAD",
    )

    return (
        base_sha,
        agent_commit_sha,
    )


def calculate_sha256(
    content: str,
) -> str:

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def create_handoff(
    *,
    repository: Path,
    branch_name: str,
    base_commit_sha: str,
    agent_commit_sha: str,
) -> GitHandoffRecord:
    """
    创建一个合法 GitHandoffRecord。

    这里测试的是 Push Service，
    所以不需要重新跑：
    Analysis / Proposal / Verification / HandoffService。

    我们直接构造 PushService 所需要的输入。
    """

    before_content = "export const value = 1\n"

    after_content = "export const value = 2\n"

    task_id = "TASK-001"

    proposal_id = hashlib.sha256(b"proposal").hexdigest()

    workspace_id = hashlib.sha256(b"workspace").hexdigest()

    handoff_id = hashlib.sha256(b"handoff").hexdigest()

    return GitHandoffRecord(
        handoff_id=handoff_id,
        task_id=task_id,
        proposal_id=proposal_id,
        workspace_id=workspace_id,
        source_repository_root=str(repository.resolve()),
        agent_branch=branch_name,
        target_branch="main",
        base_commit_sha=(base_commit_sha),
        agent_commit_sha=(agent_commit_sha),
        integration_status=("up_to_date"),
        files=[
            GitHandoffFile(
                relative_path="user.ts",
                before_sha256=(calculate_sha256(before_content)),
                after_sha256=(calculate_sha256(after_content)),
            )
        ],
    )


def prepare_push_case(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    GitHandoffRecord,
]:
    """
    构建大多数 Push Test 共用场景：

    source repo
        ↓
    initial commit A
        ↓
    agent branch
        ↓
    agent commit D

    同时创建：
        bare remote.git
        origin remote
        GitHandoffRecord
    """

    repository = tmp_path / "repo"

    create_repository(repository)

    remote = create_bare_remote(tmp_path)

    run_git(
        repository,
        "remote",
        "add",
        "origin",
        str(remote),
    )

    branch_name = "fe-agent/TASK-001-test"

    (
        base_commit_sha,
        agent_commit_sha,
    ) = create_agent_commit(
        repository,
        branch_name=branch_name,
    )

    handoff = create_handoff(
        repository=repository,
        branch_name=branch_name,
        base_commit_sha=(base_commit_sha),
        agent_commit_sha=(agent_commit_sha),
    )

    return (
        repository,
        remote,
        handoff,
    )


# ============================================================
# Tests
# ============================================================


@pytest.mark.asyncio
async def test_push_agent_commit(
    tmp_path: Path,
):
    """
    正常场景：

    local:
        agent branch -> D

    remote:
        还没有 agent branch

    预期：
        push 成功
        remote branch -> D
    """

    (
        repository,
        remote,
        handoff,
    ) = prepare_push_case(tmp_path)

    service = AgentGitPushService()

    result = await service.push(
        handoff=handoff,
        remote_name="origin",
    )

    assert result.handoff_id == handoff.handoff_id

    assert result.task_id == handoff.task_id

    assert result.proposal_id == handoff.proposal_id

    assert result.remote_name == "origin"

    assert result.branch_name == handoff.agent_branch

    assert result.commit_sha == handoff.agent_commit_sha

    remote_sha = run_bare_git(
        remote,
        "rev-parse",
        ("refs/heads/" + handoff.agent_branch),
    )

    assert remote_sha == handoff.agent_commit_sha


@pytest.mark.asyncio
async def test_push_is_idempotent(
    tmp_path: Path,
):
    """
    第一次 Push 已成功后，
    第二次用同一个 Handoff 再 Push。

    远端已经是同一个 Commit，
    应直接返回成功，而不是报错。
    """

    (
        repository,
        remote,
        handoff,
    ) = prepare_push_case(tmp_path)

    service = AgentGitPushService()

    first = await service.push(
        handoff=handoff,
        remote_name="origin",
    )

    second = await service.push(
        handoff=handoff,
        remote_name="origin",
    )

    assert first.commit_sha == handoff.agent_commit_sha

    assert second.commit_sha == handoff.agent_commit_sha

    remote_sha = run_bare_git(
        remote,
        "rev-parse",
        ("refs/heads/" + handoff.agent_branch),
    )

    assert remote_sha == handoff.agent_commit_sha


@pytest.mark.asyncio
async def test_push_rejects_missing_remote(
    tmp_path: Path,
):
    """
    Handoff 合法，
    但指定的 Git Remote 不存在。

    必须拒绝。
    """

    (
        repository,
        remote,
        handoff,
    ) = prepare_push_case(tmp_path)

    service = AgentGitPushService()

    with pytest.raises(
        GitPushError,
        match="Git Remote 不存在",
    ):
        await service.push(
            handoff=handoff,
            remote_name="upstream",
        )


@pytest.mark.asyncio
async def test_push_rejects_empty_remote_name(
    tmp_path: Path,
):
    """
    remote_name 不能为空。
    """

    (
        repository,
        remote,
        handoff,
    ) = prepare_push_case(tmp_path)

    service = AgentGitPushService()

    with pytest.raises(
        GitPushError,
        match="remote_name 不能为空",
    ):
        await service.push(
            handoff=handoff,
            remote_name="   ",
        )


@pytest.mark.asyncio
async def test_push_rejects_when_local_branch_advanced(
    tmp_path: Path,
):
    """
    Handoff 时：

        agent branch -> D

    后来又发生：

        D -> E

    Handoff 仍然只批准 D。

    预期：
        Push 拒绝。

    原因：
        Handoff 之后本地 Agent Branch
        已经发生变化。
    """

    (
        repository,
        remote,
        handoff,
    ) = prepare_push_case(tmp_path)

    (repository / "later.ts").write_text(
        "export const later = true\n",
        encoding="utf-8",
    )

    run_git(
        repository,
        "add",
        "--all",
    )

    run_git(
        repository,
        "commit",
        "-m",
        "later agent change",
    )

    current_head = run_git(
        repository,
        "rev-parse",
        "HEAD",
    )

    assert current_head != handoff.agent_commit_sha

    service = AgentGitPushService()

    with pytest.raises(
        GitPushError,
        match="Agent Branch HEAD 已经发生变化",
    ):
        await service.push(
            handoff=handoff,
            remote_name="origin",
        )


@pytest.mark.asyncio
async def test_push_rejects_when_remote_branch_has_different_commit(
    tmp_path: Path,
):
    """
    远端已经存在同名 Agent Branch，
    但是它指向另一个 Commit。

    当前策略：
        不 Force Push
        不覆盖
        直接拒绝。
    """

    (
        repository,
        remote,
        handoff,
    ) = prepare_push_case(tmp_path)

    # ----------------------------------
    # 创建一个不同的 remote commit。
    #
    # 这里新建临时 clone，
    # 模拟“另一个人已经把同名 branch
    # 推到了远端”。
    # ----------------------------------

    other = tmp_path / "other"

    subprocess.run(
        [
            "git",
            "clone",
            str(remote),
            str(other),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    run_git(
        other,
        "config",
        "user.email",
        "other@example.com",
    )

    run_git(
        other,
        "config",
        "user.name",
        "Other User",
    )

    # remote 目前还是空仓库，
    # 所以先创建一个完全独立的提交。
    (other / "other.txt").write_text(
        "other change\n",
        encoding="utf-8",
    )

    run_git(
        other,
        "add",
        "--all",
    )

    run_git(
        other,
        "commit",
        "-m",
        "other commit",
    )

    other_sha = run_git(
        other,
        "rev-parse",
        "HEAD",
    )

    assert other_sha != handoff.agent_commit_sha

    run_git(
        other,
        "push",
        "origin",
        (f"{other_sha}:" f"refs/heads/" f"{handoff.agent_branch}"),
    )

    remote_sha_before = run_bare_git(
        remote,
        "rev-parse",
        ("refs/heads/" + handoff.agent_branch),
    )

    assert remote_sha_before == other_sha

    # ----------------------------------
    # Agent 尝试 Push 自己的 Handoff。
    # ----------------------------------

    service = AgentGitPushService()

    with pytest.raises(
        GitPushError,
        match=("远端 Agent Branch " "已经存在不同 Commit"),
    ):
        await service.push(
            handoff=handoff,
            remote_name="origin",
        )

    # ----------------------------------
    # 最关键：
    #
    # Agent 失败后不能覆盖 remote。
    # ----------------------------------

    remote_sha_after = run_bare_git(
        remote,
        "rev-parse",
        ("refs/heads/" + handoff.agent_branch),
    )

    assert remote_sha_after == other_sha


@pytest.mark.asyncio
async def test_push_rejects_missing_repository(
    tmp_path: Path,
):
    """
    Handoff 指向的 source repository
    已经不存在。

    Push 必须拒绝。
    """

    fake_repository = tmp_path / "missing-repo"

    handoff = GitHandoffRecord(
        handoff_id=(hashlib.sha256(b"handoff").hexdigest()),
        task_id="TASK-001",
        proposal_id=(hashlib.sha256(b"proposal").hexdigest()),
        workspace_id=(hashlib.sha256(b"workspace").hexdigest()),
        source_repository_root=str(fake_repository),
        agent_branch=("fe-agent/TASK-001-test"),
        target_branch="main",
        base_commit_sha="a" * 40,
        agent_commit_sha="b" * 40,
        integration_status=("up_to_date"),
        files=[
            GitHandoffFile(
                relative_path="user.ts",
                before_sha256="a" * 64,
                after_sha256="b" * 64,
            )
        ],
    )

    service = AgentGitPushService()

    with pytest.raises(
        GitPushError,
        match="Source Repository 不存在",
    ):
        await service.push(handoff=handoff)
