from pathlib import Path

import pytest

from agent_core.coding.checkpoint import (
    AnalysisCheckpointStore,
)
from agent_core.coding.handoff import (
    AnalysisArtifactStore,
)
from agent_core.coding.model import (
    ReadOnlyCodingResult,
)
from agent_core.coding.orchestration_model import (
    TaskAnalysisRecord,
)
from tests.test_context_renderer import (
    create_test_bundle,
)


def make_result(
    task_id: str = "TASK-003",
) -> ReadOnlyCodingResult:
    return ReadOnlyCodingResult(
        task_id=task_id,
        summary="已完成只读代码分析",
        findings=[],
        proposed_changes=[],
        verification_plan=["后续验证查询参数"],
        missing_information=[],
        execution_status="not_executed",
    )


def make_success_record() -> TaskAnalysisRecord:
    return TaskAnalysisRecord(
        task_id="TASK-003",
        status="succeeded",
        attempts=2,
        result=make_result(),
    )


def test_checkpoint_preserves_attempts(
    tmp_path: Path,
):
    store = AnalysisCheckpointStore(tmp_path)

    store.save({"TASK-003": make_success_record()})

    loaded = store.load()

    record = loaded.records["TASK-003"]

    assert record.status == "succeeded"
    assert record.attempts == 2
    assert record.result is not None


def test_checkpoint_survives_new_store_instance(
    tmp_path: Path,
):
    first_store = AnalysisCheckpointStore(tmp_path)

    first_store.save({"TASK-003": make_success_record()})

    # 模拟程序重启：创建全新的 Store 实例。
    second_store = AnalysisCheckpointStore(tmp_path)

    checkpoint = second_store.load()

    assert checkpoint.records["TASK-003"].attempts == 2


def test_empty_checkpoint_is_valid(
    tmp_path: Path,
):
    store = AnalysisCheckpointStore(tmp_path)

    checkpoint = store.load()

    assert checkpoint.records == {}


def test_artifact_publish_and_load(
    tmp_path: Path,
):
    bundle = create_test_bundle(tmp_path)

    store = AnalysisArtifactStore(tmp_path)

    artifact, path = store.publish(
        bundle=bundle,
        record=make_success_record(),
    )

    assert path.exists()

    assert artifact.task_id == "TASK-003"

    assert artifact.status == "proposed"

    assert artifact.result.execution_status == "not_executed"

    restored = store.load(artifact.artifact_id)

    assert restored == artifact


def test_artifact_id_is_stable(
    tmp_path: Path,
):
    bundle = create_test_bundle(tmp_path)

    record = make_success_record()

    store = AnalysisArtifactStore(tmp_path)

    first, _ = store.publish(
        bundle=bundle,
        record=record,
    )

    second, _ = store.publish(
        bundle=bundle,
        record=record,
    )

    assert first.artifact_id == second.artifact_id


def test_failed_analysis_cannot_be_published(
    tmp_path: Path,
):
    store = AnalysisArtifactStore(tmp_path)

    failed_record = TaskAnalysisRecord(
        task_id="TASK-003",
        status="failed",
        attempts=1,
        error="模拟分析失败",
    )

    with pytest.raises(
        ValueError,
        match="成功",
    ):
        store.publish(
            bundle=create_test_bundle(tmp_path),
            record=failed_record,
        )


def test_invalid_artifact_id_is_rejected(
    tmp_path: Path,
):
    store = AnalysisArtifactStore(tmp_path)

    with pytest.raises(
        ValueError,
        match="非法 Artifact ID",
    ):
        store.load("../../secrets")
