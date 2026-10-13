from pathlib import Path

import pytest

from agent_core.git.follow_up_task import (
    FollowUpTaskError,
    FollowUpTaskGenerator,
)
from agent_core.git.follow_up_task_store import (
    FollowUpTaskStore,
)
from agent_core.git.review_feedback_analysis_model import (
    ReviewFeedbackAnalysisArtifact,
    ReviewFeedbackAnalysisItem,
)
from agent_core.git.review_feedback_model import (
    ReviewFeedbackArtifact,
    ReviewFeedbackItem,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


def create_parent_task() -> TaskDefinition:

    return TaskDefinition(
        task_id="TASK-001",
        task_type="code_change",
        title="实现用户查询",
        objective=("实现用户列表查询逻辑"),
        source_step_order=2,
        depends_on=[],
        inputs=["Implementation Plan Step 2"],
        target_modules=["user"],
        expected_outputs=["用户查询逻辑完成"],
        verification=["相关测试通过"],
        acceptance_criteria_refs=["用户可以查询用户列表"],
    )


def create_feedback() -> ReviewFeedbackArtifact:

    return ReviewFeedbackArtifact(
        artifact_id="a" * 64,
        pull_request_provider="github",
        repository=("coding-man-wang/" "fe-agent"),
        pull_request_number=123,
        task_id="TASK-001",
        proposal_id="b" * 64,
        commit_sha="c" * 40,
        items=[
            ReviewFeedbackItem(
                feedback_id="d" * 64,
                kind="inline_comment",
                reviewer="alice",
                body=("这里不要直接调用 fetch，" "请复用 requestClient"),
                relative_path=("src/user.ts"),
                line=42,
                commit_sha="c" * 40,
                external_id="101",
                url=None,
            ),
            ReviewFeedbackItem(
                feedback_id="e" * 64,
                kind=("conversation_comment"),
                reviewer="bob",
                body=("为什么这里不用缓存？"),
                relative_path=None,
                line=None,
                commit_sha=None,
                external_id="102",
                url=None,
            ),
            ReviewFeedbackItem(
                feedback_id="f" * 64,
                kind=("conversation_comment"),
                reviewer="charlie",
                body="LGTM",
                relative_path=None,
                line=None,
                commit_sha=None,
                external_id="103",
                url=None,
            ),
        ],
    )


def create_analysis() -> ReviewFeedbackAnalysisArtifact:

    return ReviewFeedbackAnalysisArtifact(
        analysis_id="9" * 64,
        source_artifact_id=("a" * 64),
        task_id="TASK-001",
        proposal_id="b" * 64,
        commit_sha="c" * 40,
        items=[
            ReviewFeedbackAnalysisItem(
                feedback_id="d" * 64,
                intent="code_change",
                summary=("复用 requestClient"),
                reasoning=("Reviewer 明确要求" "修改请求实现方式"),
                requires_code_change=True,
                requires_reply=False,
                target_hints=[
                    "src/user.ts",
                    "requestClient",
                    "requestClient",
                ],
            ),
            ReviewFeedbackAnalysisItem(
                feedback_id="e" * 64,
                intent="discussion",
                summary=("解释缓存设计"),
                reasoning=("Reviewer 提出设计问题"),
                requires_code_change=False,
                requires_reply=True,
                target_hints=["cache"],
            ),
            ReviewFeedbackAnalysisItem(
                feedback_id="f" * 64,
                intent="non_actionable",
                summary="无需处理",
                reasoning="LGTM",
                requires_code_change=False,
                requires_reply=False,
                target_hints=[],
            ),
        ],
    )


def test_generates_task_only_for_code_change():

    generator = FollowUpTaskGenerator()

    artifact = generator.generate(
        parent_task=(create_parent_task()),
        feedback=create_feedback(),
        analysis=create_analysis(),
    )

    assert len(artifact.tasks) == 1

    task = artifact.tasks[0]

    assert task.task_type == "code_change"

    assert task.task_id.startswith("TASK-001-REV-")

    assert "复用 requestClient" in task.title

    assert task.source_step_order == 2

    assert task.acceptance_criteria_refs == ["用户可以查询用户列表"]

    assert task.target_modules == [
        "src/user.ts",
        "requestClient",
    ]

    assert task.depends_on == []

    assert len(artifact.links) == 1

    assert artifact.links[0].feedback_id == "d" * 64

    assert artifact.links[0].follow_up_task_id == task.task_id


def test_does_not_generate_task_for_discussion():

    analysis = create_analysis()

    analysis = analysis.model_copy(
        update={
            "items": [item for item in analysis.items if (item.intent != "code_change")]
        }
    )

    feedback = create_feedback()

    feedback = feedback.model_copy(
        update={
            "items": [item for item in feedback.items if (item.feedback_id != "d" * 64)]
        }
    )

    generator = FollowUpTaskGenerator()

    artifact = generator.generate(
        parent_task=(create_parent_task()),
        feedback=feedback,
        analysis=analysis,
    )

    assert artifact.tasks == []

    assert artifact.links == []


def test_task_generation_is_deterministic():

    generator = FollowUpTaskGenerator()

    kwargs = {
        "parent_task": create_parent_task(),
        "feedback": create_feedback(),
        "analysis": create_analysis(),
    }

    first = generator.generate(**kwargs)

    second = generator.generate(**kwargs)

    assert first.artifact_id == second.artifact_id

    assert first.tasks[0].task_id == second.tasks[0].task_id


def test_rejects_analysis_for_wrong_parent_task():

    analysis = create_analysis().model_copy(update={"task_id": "TASK-999"})

    generator = FollowUpTaskGenerator()

    with pytest.raises(
        FollowUpTaskError,
        match="parent_task",
    ):
        generator.generate(
            parent_task=(create_parent_task()),
            feedback=create_feedback(),
            analysis=analysis,
        )


def test_rejects_wrong_feedback_artifact_binding():

    analysis = create_analysis().model_copy(update={"source_artifact_id": "8" * 64})

    generator = FollowUpTaskGenerator()

    with pytest.raises(
        FollowUpTaskError,
        match=("Artifact 不匹配"),
    ):
        generator.generate(
            parent_task=(create_parent_task()),
            feedback=create_feedback(),
            analysis=analysis,
        )


def test_rejects_commit_mismatch():

    analysis = create_analysis().model_copy(update={"commit_sha": "7" * 40})

    generator = FollowUpTaskGenerator()

    with pytest.raises(
        FollowUpTaskError,
        match="commit_sha",
    ):
        generator.generate(
            parent_task=(create_parent_task()),
            feedback=create_feedback(),
            analysis=analysis,
        )


def test_rejects_feedback_analysis_scope_mismatch():

    analysis = create_analysis()

    analysis = analysis.model_copy(update={"items": analysis.items[:-1]})

    generator = FollowUpTaskGenerator()

    with pytest.raises(
        FollowUpTaskError,
        match="范围不一致",
    ):
        generator.generate(
            parent_task=(create_parent_task()),
            feedback=create_feedback(),
            analysis=analysis,
        )


def test_rejects_inconsistent_code_change():

    analysis = create_analysis()

    broken_item = analysis.items[0].model_copy(update={"requires_code_change": False})

    analysis = analysis.model_copy(
        update={
            "items": [
                broken_item,
                *analysis.items[1:],
            ]
        }
    )

    generator = FollowUpTaskGenerator()

    with pytest.raises(
        FollowUpTaskError,
        match="必须要求代码修改",
    ):
        generator.generate(
            parent_task=(create_parent_task()),
            feedback=create_feedback(),
            analysis=analysis,
        )


def test_store_round_trip(
    tmp_path: Path,
):

    repository = tmp_path / "repository"

    repository.mkdir()

    generator = FollowUpTaskGenerator()

    artifact = generator.generate(
        parent_task=(create_parent_task()),
        feedback=create_feedback(),
        analysis=create_analysis(),
    )

    store = FollowUpTaskStore(repository)

    store.save(artifact)

    loaded = store.load(artifact.artifact_id)

    assert loaded == artifact
