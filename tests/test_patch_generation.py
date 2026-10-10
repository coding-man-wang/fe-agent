import hashlib
import json
from pathlib import Path

import pytest

from agent_core.coding.handoff import (
    AnalysisArtifact,
)
from agent_core.coding.model import (
    ProposedCodeChange,
    ReadOnlyCodingResult,
)
from agent_core.coding.patch_generator import (
    PatchGenerationError,
    PatchGenerator,
)
from agent_core.coding.patch_model import (
    PatchPlan,
    PatchTarget,
)
from agent_core.coding.preparation_model import (
    ExecutionPreparation,
)

BEFORE = "export function getUserList() {\n" "    return fetch('/api/users')\n" "}\n"

AFTER = (
    "export function getUserList(status: string) {\n"
    "    return fetch(`/api/users?status=${status}`)\n"
    "}\n"
)


class FakePatchModel:

    def __init__(
        self,
        *,
        response: dict,
    ) -> None:
        self.response = response
        self.calls = 0
        self.last_user_message: str | None = None

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str:
        self.calls += 1

        self.last_user_message = user_message

        return json.dumps(
            self.response,
            ensure_ascii=False,
        )


def sha256_text(
    text: str,
) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_model_response(
    *,
    path: str = "src/api/user.ts",
    after_content: str = AFTER,
) -> dict:
    return {
        "task_id": "TASK-003",
        "changes": [
            {
                "relative_path": path,
                "after_content": after_content,
                "explanation": "增加 status 查询参数",
            }
        ],
    }


def create_environment(
    root: Path,
    *,
    model_response: dict | None = None,
):
    relative_path = "src/api/user.ts"

    file_path = root / relative_path

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path.write_bytes(BEFORE.encode("utf-8"))

    before_hash = sha256_text(BEFORE)

    result = ReadOnlyCodingResult(
        task_id="TASK-003",
        summary="扩展用户列表 API",
        findings=[],
        proposed_changes=[
            ProposedCodeChange(
                relative_path=relative_path,
                objective="支持状态筛选",
                proposed_approach=("为 getUserList 增加 status 参数"),
            )
        ],
        verification_plan=["验证请求参数"],
        missing_information=[],
        execution_status="not_executed",
    )

    artifact = AnalysisArtifact(
        artifact_id="a" * 64,
        task_id="TASK-003",
        context_hash="b" * 64,
        result=result,
    )

    plan = PatchPlan(
        task_id="TASK-003",
        artifact_id=artifact.artifact_id,
        context_hash=artifact.context_hash,
        targets=[
            PatchTarget(
                relative_path=relative_path,
                action="modify",
                expected_sha256=before_hash,
                objective="增加 status 参数",
                proposed_approach=("修改用户查询函数"),
            )
        ],
    )

    preparation = ExecutionPreparation(
        task_id="TASK-003",
        artifact_id=artifact.artifact_id,
        context_hash=artifact.context_hash,
        proposed_paths=[relative_path],
        checked_context_paths=[relative_path],
        checks_passed=[
            "artifact_integrity",
            "task_consistency",
            "context_consistency",
            "repository_freshness",
            "path_boundary",
        ],
    )

    model = FakePatchModel(
        response=(
            model_response if model_response is not None else make_model_response()
        )
    )

    generator = PatchGenerator(
        repository_root=root,
        model=model,
    )

    return (
        artifact,
        plan,
        preparation,
        model,
        generator,
        file_path,
    )


@pytest.mark.asyncio
async def test_generate_valid_patch_without_writing(
    tmp_path: Path,
):
    (
        artifact,
        plan,
        preparation,
        model,
        generator,
        file_path,
    ) = create_environment(tmp_path)

    before_bytes = file_path.read_bytes()

    proposal = await generator.generate(
        artifact=artifact,
        plan=plan,
        preparation=preparation,
    )

    assert proposal.status == "generated_not_applied"

    assert len(proposal.patches) == 1

    patch = proposal.patches[0]

    assert patch.before_content == BEFORE
    assert patch.after_content == AFTER

    assert "status: string" in (patch.unified_diff)

    assert patch.expected_sha256 == (sha256_text(BEFORE))

    assert patch.after_sha256 == (sha256_text(AFTER))

    # 最重要的断言：生成阶段没有落盘。
    assert file_path.read_bytes() == before_bytes

    assert model.calls == 1


@pytest.mark.asyncio
async def test_rejects_unplanned_file(
    tmp_path: Path,
):
    (
        artifact,
        plan,
        preparation,
        _,
        generator,
        file_path,
    ) = create_environment(
        tmp_path,
        model_response=make_model_response(path="src/main.ts"),
    )

    with pytest.raises(
        PatchGenerationError,
        match="修改范围",
    ):
        await generator.generate(
            artifact=artifact,
            plan=plan,
            preparation=preparation,
        )

    assert file_path.read_text(encoding="utf-8") == BEFORE


@pytest.mark.asyncio
async def test_rejects_noop_patch(
    tmp_path: Path,
):
    (
        artifact,
        plan,
        preparation,
        _,
        generator,
        _,
    ) = create_environment(
        tmp_path,
        model_response=make_model_response(after_content=BEFORE),
    )

    with pytest.raises(
        PatchGenerationError,
        match="没有产生实际修改",
    ):
        await generator.generate(
            artifact=artifact,
            plan=plan,
            preparation=preparation,
        )


@pytest.mark.asyncio
async def test_rejects_changed_file_before_model_call(
    tmp_path: Path,
):
    (
        artifact,
        plan,
        preparation,
        model,
        generator,
        file_path,
    ) = create_environment(tmp_path)

    file_path.write_text(
        "export function changed() {}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        PatchGenerationError,
        match="版本已经变化",
    ):
        await generator.generate(
            artifact=artifact,
            plan=plan,
            preparation=preparation,
        )

    assert model.calls == 0


class MutatingPatchModel:

    def __init__(
        self,
        file_path: Path,
    ) -> None:
        self.file_path = file_path

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str:
        # 模拟模型调用期间其他程序修改文件。
        self.file_path.write_text(
            "export function externalChange() {}\n",
            encoding="utf-8",
        )

        return json.dumps(
            make_model_response(),
            ensure_ascii=False,
        )


@pytest.mark.asyncio
async def test_detects_change_during_generation(
    tmp_path: Path,
):
    (
        artifact,
        plan,
        preparation,
        _,
        _,
        file_path,
    ) = create_environment(tmp_path)

    generator = PatchGenerator(
        repository_root=tmp_path,
        model=MutatingPatchModel(file_path),
    )

    with pytest.raises(
        PatchGenerationError,
        match="版本已经变化",
    ):
        await generator.generate(
            artifact=artifact,
            plan=plan,
            preparation=preparation,
        )
