from pathlib import Path

import pytest

from agent_core.coding.handoff import (
    AnalysisArtifactStore,
    calculate_messages_hash,
)
from agent_core.coding.model import (
    ProposedCodeChange,
    ReadOnlyCodingResult,
)
from agent_core.coding.orchestration_model import (
    TaskAnalysisRecord,
)
from agent_core.coding.preparation import (
    ExecutionPreparer,
    PreparationError,
)
from agent_core.context.model import (
    RenderedContext,
    TaskContextBundle,
)
from agent_core.context.renderer import (
    ContextRenderer,
)
from tests.test_context_renderer import (
    create_test_bundle,
)


class FakeMessageProvider:

    def prepare_messages(
        self,
        bundle: TaskContextBundle,
    ) -> RenderedContext:
        rendered = ContextRenderer().render(bundle)

        return RenderedContext(
            system_message=(rendered.system_message + "\n\n只读分析输出规范"),
            user_message=rendered.user_message,
        )


def create_environment(
    root: Path,
    *,
    proposed_path: str = "src/api/user.ts",
):
    bundle = create_test_bundle(root)

    code_item = bundle.code_context[0]

    file_path = root / code_item.relative_path

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path.write_text(
        code_item.content,
        encoding="utf-8",
    )

    result = ReadOnlyCodingResult(
        task_id=bundle.task.task_id,
        summary="建议增加 status 参数",
        findings=[],
        proposed_changes=[
            ProposedCodeChange(
                relative_path=proposed_path,
                objective="扩展查询参数",
                proposed_approach=("在现有 API 中增加 status"),
            )
        ],
        verification_plan=["验证 status 参数传递"],
        missing_information=[],
        execution_status="not_executed",
    )

    record = TaskAnalysisRecord(
        task_id=bundle.task.task_id,
        status="succeeded",
        attempts=1,
        result=result,
    )

    provider = FakeMessageProvider()

    messages = provider.prepare_messages(bundle)

    context_hash = calculate_messages_hash(messages)

    store = AnalysisArtifactStore(root)

    artifact, _ = store.publish(
        bundle=bundle,
        record=record,
        context_hash=context_hash,
    )

    preparer = ExecutionPreparer(
        repository_root=root,
        artifact_store=store,
        message_provider=provider,
    )

    return (
        bundle,
        artifact,
        preparer,
        file_path,
    )


def test_valid_artifact_is_ready_for_review(
    tmp_path: Path,
):
    bundle, artifact, preparer, _ = create_environment(tmp_path)

    preparation = preparer.prepare(
        task_id="TASK-003",
        artifact_id=artifact.artifact_id,
        bundle=bundle,
    )

    assert preparation.status == "ready_for_review"

    assert preparation.proposed_paths == ["src/api/user.ts"]

    assert "repository_freshness" in preparation.checks_passed


def test_rejects_wrong_task_id(
    tmp_path: Path,
):
    bundle, artifact, preparer, _ = create_environment(tmp_path)

    with pytest.raises(
        PreparationError,
        match="当前 Task",
    ):
        preparer.prepare(
            task_id="TASK-999",
            artifact_id=artifact.artifact_id,
            bundle=bundle,
        )


def test_rejects_changed_file(
    tmp_path: Path,
):
    bundle, artifact, preparer, file_path = create_environment(tmp_path)

    # 模拟其他开发者修改了文件。
    file_path.write_text(
        "export function getUserList(status) {}",
        encoding="utf-8",
    )

    with pytest.raises(
        PreparationError,
        match="发生变化",
    ):
        preparer.prepare(
            task_id="TASK-003",
            artifact_id=artifact.artifact_id,
            bundle=bundle,
        )


def test_rejects_path_traversal(
    tmp_path: Path,
):
    bundle, artifact, preparer, _ = create_environment(
        tmp_path,
        proposed_path="../../outside.ts",
    )

    with pytest.raises(
        PreparationError,
        match="路径穿越",
    ):
        preparer.prepare(
            task_id="TASK-003",
            artifact_id=artifact.artifact_id,
            bundle=bundle,
        )


def test_rejects_unprovided_target_file(
    tmp_path: Path,
):
    bundle, artifact, preparer, _ = create_environment(
        tmp_path,
        proposed_path="src/api/unknown.ts",
    )

    with pytest.raises(
        PreparationError,
        match="不在已验证的 Context",
    ):
        preparer.prepare(
            task_id="TASK-003",
            artifact_id=artifact.artifact_id,
            bundle=bundle,
        )


def test_rejects_changed_context(
    tmp_path: Path,
):
    bundle, artifact, preparer, _ = create_environment(tmp_path)

    changed_requirement = bundle.requirement.model_copy(
        update={"summary": "已经变更的需求"}
    )

    changed_bundle = bundle.model_copy(update={"requirement": changed_requirement})

    with pytest.raises(
        PreparationError,
        match="Context Hash",
    ):
        preparer.prepare(
            task_id="TASK-003",
            artifact_id=artifact.artifact_id,
            bundle=changed_bundle,
        )


def test_rejects_tampered_artifact(
    tmp_path: Path,
):
    bundle, artifact, preparer, _ = create_environment(tmp_path)

    store = AnalysisArtifactStore(tmp_path)

    path = store.artifact_dir / f"{artifact.artifact_id}.json"

    import json

    data = json.loads(path.read_text(encoding="utf-8"))

    data["result"]["summary"] = "被修改过的分析内容"

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        PreparationError,
        match="完整性",
    ):
        preparer.prepare(
            task_id="TASK-003",
            artifact_id=artifact.artifact_id,
            bundle=bundle,
        )
