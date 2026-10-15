import json
from pathlib import Path

import pytest

from agent_core.benchmark.case_loader import (
    BenchmarkCaseLoadError,
    BenchmarkCaseLoader,
)
from agent_core.benchmark.repository_guard import (
    BenchmarkRepositoryError,
    BenchmarkRepositoryGuard,
)
from agent_core.git.client import (
    GitCommandResult,
)

BASE_SHA = "a" * 40


def write_case(
    root: Path,
    *,
    case_id: str = "CASE-001",
    base_commit_sha: str = BASE_SHA,
) -> None:

    data = {
        "schema_version": 1,
        "case_id": case_id,
        "title": "test",
        "category": ("single_file_change"),
        "difficulty": "easy",
        "repository_name": ("benchmark-web"),
        "base_commit_sha": (base_commit_sha),
        "requirement": ("修改用户名长度限制"),
        "expectation": {
            "expected_files": ["src/components/UserProfileForm.vue"],
            "forbidden_files": ["src/api"],
            "expected_behaviors": ["最大长度为 30"],
            "verification_command_ids": ["frontend-unit-test"],
        },
    }

    path = root / f"{case_id}.json"

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_case_loader_loads_valid_case(
    tmp_path: Path,
):

    case_root = tmp_path / "cases"

    case_root.mkdir()

    write_case(case_root)

    loader = BenchmarkCaseLoader(case_root)

    case = loader.load("CASE-001")

    assert case.case_id == "CASE-001"

    assert case.base_commit_sha == BASE_SHA

    assert case.category == "single_file_change"

    assert case.expectation.expected_files == ["src/components/UserProfileForm.vue"]


def test_case_loader_rejects_path_traversal(
    tmp_path: Path,
):

    case_root = tmp_path / "cases"

    case_root.mkdir()

    loader = BenchmarkCaseLoader(case_root)

    with pytest.raises(
        BenchmarkCaseLoadError,
        match="非法 case_id",
    ):

        loader.load("../CASE-001")


def test_case_loader_rejects_invalid_case(
    tmp_path: Path,
):

    case_root = tmp_path / "cases"

    case_root.mkdir()

    path = case_root / "CASE-001.json"

    path.write_text(
        "{}",
        encoding="utf-8",
    )

    loader = BenchmarkCaseLoader(case_root)

    with pytest.raises(
        BenchmarkCaseLoadError,
        match="Benchmark Case 非法",
    ):

        loader.load("CASE-001")


def test_case_loader_rejects_filename_mismatch(
    tmp_path: Path,
):

    case_root = tmp_path / "cases"

    case_root.mkdir()

    write_case(
        case_root,
        case_id="CASE-002",
    )

    source = case_root / "CASE-002.json"

    target = case_root / "CASE-001.json"

    source.rename(target)

    loader = BenchmarkCaseLoader(case_root)

    with pytest.raises(
        BenchmarkCaseLoadError,
        match=("文件名与 case_id 不一致"),
    ):

        loader.load("CASE-001")


class FakeGitClient:

    def __init__(
        self,
        *,
        head: str = BASE_SHA,
        status: str = "",
    ) -> None:

        self.head = head
        self.status = status
        self.calls: list[list[str]] = []

    async def run(
        self,
        *,
        cwd: Path,
        args: list[str],
        allow_failure: bool = False,
    ) -> GitCommandResult:

        self.calls.append(list(args))

        if args == [
            "rev-parse",
            "HEAD",
        ]:

            return GitCommandResult(
                exit_code=0,
                stdout=(self.head + "\n"),
                stderr="",
            )

        if args == [
            "status",
            "--porcelain",
            "--untracked-files=all",
        ]:

            return GitCommandResult(
                exit_code=0,
                stdout=self.status,
                stderr="",
            )

        raise AssertionError(f"未预期 Git 命令: {args}")


def create_case():
    from agent_core.benchmark.case_model import (
        BenchmarkCase,
    )

    return BenchmarkCase(
        case_id="CASE-001",
        title="test",
        category=("single_file_change"),
        difficulty="easy",
        repository_name=("benchmark-web"),
        base_commit_sha=BASE_SHA,
        requirement="test",
        expectation={
            "expected_files": ["src/a.vue"],
            "forbidden_files": [],
            "expected_behaviors": ["works"],
            "verification_command_ids": ["test"],
        },
    )


@pytest.mark.asyncio
async def test_repository_guard_accepts_ready_repository(
    tmp_path: Path,
):

    repository = tmp_path / "repo"

    repository.mkdir()

    guard = BenchmarkRepositoryGuard(
        repository_root=repository,
        git_client=FakeGitClient(),
    )

    state = await guard.require_ready(case=create_case())

    assert state.actual_commit_sha == BASE_SHA

    assert state.expected_commit_sha == BASE_SHA

    assert state.clean is True


@pytest.mark.asyncio
async def test_repository_guard_rejects_wrong_head(
    tmp_path: Path,
):

    repository = tmp_path / "repo"

    repository.mkdir()

    guard = BenchmarkRepositoryGuard(
        repository_root=repository,
        git_client=(FakeGitClient(head="b" * 40)),
    )

    with pytest.raises(
        BenchmarkRepositoryError,
        match="Base Commit 不一致",
    ):

        await guard.require_ready(case=create_case())


@pytest.mark.asyncio
async def test_repository_guard_rejects_dirty_repository(
    tmp_path: Path,
):

    repository = tmp_path / "repo"

    repository.mkdir()

    guard = BenchmarkRepositoryGuard(
        repository_root=repository,
        git_client=(FakeGitClient(status=(" M src/a.vue\n"))),
    )

    with pytest.raises(
        BenchmarkRepositoryError,
        match="存在未提交修改",
    ):

        await guard.require_ready(case=create_case())
