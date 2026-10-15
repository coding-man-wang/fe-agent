import json
from pathlib import Path

import pytest

from agent_core.benchmark.case_loader import (
    BenchmarkCaseLoader,
)
from agent_core.benchmark.repository_guard import (
    BenchmarkRepositoryGuard,
)
from agent_core.benchmark.run_store import (
    BenchmarkRunStore,
)
from agent_core.benchmark.runner import (
    BenchmarkRunner,
)
from agent_core.git.client import (
    GitCommandResult,
)
from agent_core.sdd.model import (
    RequirementSpec,
)

BASE_SHA = "a" * 40


def write_case(
    root: Path,
) -> None:

    data = {
        "schema_version": 1,
        "case_id": "CASE-001",
        "title": "test",
        "category": ("single_file_change"),
        "difficulty": "easy",
        "repository_name": ("benchmark-web"),
        "base_commit_sha": (BASE_SHA),
        "requirement": ("用户名最大长度修改为 30"),
        "expectation": {
            "expected_files": ["src/components/" "UserProfileForm.vue"],
            "forbidden_files": ["src/api"],
            "expected_behaviors": ["最大长度为 30"],
            "verification_command_ids": ["frontend-unit-test"],
        },
    }

    (root / "CASE-001.json").write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class FakeGitClient:

    async def run(
        self,
        *,
        cwd: Path,
        args: list[str],
        allow_failure: bool = False,
    ) -> GitCommandResult:

        if args == [
            "rev-parse",
            "HEAD",
        ]:

            return GitCommandResult(
                exit_code=0,
                stdout=(BASE_SHA + "\n"),
                stderr="",
            )

        if args == [
            "status",
            "--porcelain",
            "--untracked-files=all",
        ]:

            return GitCommandResult(
                exit_code=0,
                stdout="",
                stderr="",
            )

        raise AssertionError(f"未预期 Git 命令: {args}")


class FakeRequirementAnalyzer:

    def __init__(
        self,
        requirement_spec: RequirementSpec,
    ) -> None:

        self.requirement_spec = requirement_spec

        self.received_requirement = None

    async def analyze(
        self,
        requirement: str,
    ) -> RequirementSpec:

        self.received_requirement = requirement

        return self.requirement_spec


class FailingRequirementAnalyzer:

    async def analyze(
        self,
        requirement: str,
    ) -> RequirementSpec:

        raise RuntimeError("LLM unavailable")


def create_requirement_spec() -> RequirementSpec:

    # 不手写 RequirementSpec 字段，
    # 使用最小 schema-compatible 数据。
    #
    # 如果你当前 RequirementSpec
    # 有不同必填字段，
    # 把下面对象调整成当前 model 定义即可。

    schema = {
        "title": "用户名长度调整",
        "summary": ("用户名最大长度调整为 30"),
        "goals": ["支持最多 30 字符用户名"],
        "non_goals": [],
        "acceptance_criteria": ["用户名最大允许长度为 " "30 个字符"],
        "constraints": [],
        "assumptions": [],
        "open_questions": [],
    }

    return RequirementSpec.model_validate(schema)


@pytest.mark.asyncio
async def test_runner_executes_requirement_analysis(
    tmp_path: Path,
):

    case_root = tmp_path / "cases"

    case_root.mkdir()

    write_case(case_root)

    repository = tmp_path / "repo"

    repository.mkdir()

    analyzer = FakeRequirementAnalyzer(create_requirement_spec())

    runner = BenchmarkRunner(
        case_loader=(BenchmarkCaseLoader(case_root)),
        repository_guard=(
            BenchmarkRepositoryGuard(
                repository_root=(repository),
                git_client=(FakeGitClient()),
            )
        ),
        requirement_analyzer=(analyzer),
        run_store=(BenchmarkRunStore(tmp_path)),
    )

    result = await runner.run(case_id="CASE-001")

    assert result.status == "completed"

    assert result.case_id == "CASE-001"

    assert result.base_commit_sha == BASE_SHA

    assert result.requirement_spec is not None

    # 非常重要：
    # Analyzer 只能收到 requirement。
    assert analyzer.received_requirement == ("用户名最大长度修改为 30")

    assert "src/components/" "UserProfileForm.vue" not in (
        analyzer.received_requirement
    )


@pytest.mark.asyncio
async def test_runner_persists_completed_run(
    tmp_path: Path,
):

    case_root = tmp_path / "cases"

    case_root.mkdir()

    write_case(case_root)

    repository = tmp_path / "repo"

    repository.mkdir()

    store = BenchmarkRunStore(tmp_path)

    runner = BenchmarkRunner(
        case_loader=(BenchmarkCaseLoader(case_root)),
        repository_guard=(
            BenchmarkRepositoryGuard(
                repository_root=(repository),
                git_client=(FakeGitClient()),
            )
        ),
        requirement_analyzer=(FakeRequirementAnalyzer(create_requirement_spec())),
        run_store=store,
    )

    result = await runner.run(case_id="CASE-001")

    loaded = store.load(result.run_id)

    assert loaded == result


@pytest.mark.asyncio
async def test_runner_persists_failure(
    tmp_path: Path,
):

    case_root = tmp_path / "cases"

    case_root.mkdir()

    write_case(case_root)

    repository = tmp_path / "repo"

    repository.mkdir()

    store = BenchmarkRunStore(tmp_path)

    runner = BenchmarkRunner(
        case_loader=(BenchmarkCaseLoader(case_root)),
        repository_guard=(
            BenchmarkRepositoryGuard(
                repository_root=(repository),
                git_client=(FakeGitClient()),
            )
        ),
        requirement_analyzer=(FailingRequirementAnalyzer()),
        run_store=store,
    )

    result = await runner.run(case_id="CASE-001")

    assert result.status == "failed"

    assert result.error is not None

    assert "LLM unavailable" in result.error

    loaded = store.load(result.run_id)

    assert loaded.status == "failed"
