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
from agent_core.coding.patch_planner import (
    PatchPlanner,
    PatchPlanningError,
)
from agent_core.coding.snapshot import (
    FileSnapshotBuilder,
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
            user_message=(rendered.user_message),
        )


def create_environment(
    root: Path,
    *,
    proposed_path: str = ("src/api/user.ts"),
):
    bundle = create_test_bundle(root)

    for item in bundle.code_context:
        path = root / item.relative_path

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            item.content,
            encoding="utf-8",
        )

    snapshot = FileSnapshotBuilder(root).build(bundle)

    result = ReadOnlyCodingResult(
        task_id=(bundle.task.task_id),
        summary=("建议修改用户列表 API"),
        findings=[],
        proposed_changes=[
            ProposedCodeChange(
                relative_path=(proposed_path),
                objective=("增加 status 参数"),
                proposed_approach=("扩展现有请求参数"),
            )
        ],
        verification_plan=["验证 status 参数"],
        missing_information=[],
        execution_status=("not_executed"),
    )

    record = TaskAnalysisRecord(
        task_id=(bundle.task.task_id),
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

    return (
        artifact,
        snapshot,
    )


def test_build_patch_plan(
    tmp_path: Path,
):
    artifact, snapshot = create_environment(tmp_path)

    plan = PatchPlanner().plan(
        artifact=artifact,
        snapshot=snapshot,
    )

    assert plan.status == "planned"

    assert len(plan.targets) == 1

    target = plan.targets[0]

    assert target.relative_path == "src/api/user.ts"

    assert len(target.expected_sha256) == 64


def test_patch_plan_uses_snapshot_hash(
    tmp_path: Path,
):
    artifact, snapshot = create_environment(tmp_path)

    plan = PatchPlanner().plan(
        artifact=artifact,
        snapshot=snapshot,
    )

    assert plan.targets[0].expected_sha256 == snapshot.files[0].sha256


def test_unknown_patch_target_rejected(
    tmp_path: Path,
):
    artifact, snapshot = create_environment(
        tmp_path,
        proposed_path=("src/api/unknown.ts"),
    )

    with pytest.raises(
        PatchPlanningError,
        match="不存在于分析快照",
    ):
        PatchPlanner().plan(
            artifact=artifact,
            snapshot=snapshot,
        )
