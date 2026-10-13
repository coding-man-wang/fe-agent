import json
from pathlib import Path

import pytest

from agent_core.git.review_feedback_analysis import (
    ReviewFeedbackAnalysisError,
    ReviewFeedbackAnalyzer,
)
from agent_core.git.review_feedback_analysis_store import (
    ReviewFeedbackAnalysisStore,
)
from agent_core.git.review_feedback_model import (
    ReviewFeedbackArtifact,
    ReviewFeedbackItem,
)


class FakeTextModel:

    def __init__(
        self,
        response: dict,
    ) -> None:

        self.response = response

        self.calls: list[tuple[str, str]] = []

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str:

        self.calls.append(
            (
                system_message,
                user_message,
            )
        )

        return json.dumps(
            self.response,
            ensure_ascii=False,
        )


def create_artifact() -> ReviewFeedbackArtifact:

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


@pytest.mark.asyncio
async def test_analyze_review_feedback():

    artifact = create_artifact()

    model = FakeTextModel(
        {
            "source_artifact_id": artifact.artifact_id,
            "items": [
                {
                    "feedback_id": "d" * 64,
                    "intent": "code_change",
                    "summary": "复用 requestClient",
                    "reasoning": "Reviewer 明确要求修改实现方式",
                    "requires_code_change": True,
                    "requires_reply": False,
                    "target_hints": [
                        "src/user.ts",
                        "requestClient",
                    ],
                },
                {
                    "feedback_id": "e" * 64,
                    "intent": "discussion",
                    "summary": "解释未使用缓存的原因",
                    "reasoning": "Reviewer 提出了设计问题",
                    "requires_code_change": False,
                    "requires_reply": True,
                    "target_hints": [
                        "cache",
                    ],
                },
                {
                    "feedback_id": "f" * 64,
                    "intent": "non_actionable",
                    "summary": "Reviewer 表示认可",
                    "reasoning": "没有进一步行动要求",
                    "requires_code_change": False,
                    "requires_reply": False,
                    "target_hints": [],
                },
            ],
        }
    )

    analyzer = ReviewFeedbackAnalyzer(model=model)

    result = await analyzer.analyze(artifact=artifact)

    assert result.source_artifact_id == artifact.artifact_id

    assert result.task_id == artifact.task_id

    assert result.commit_sha == artifact.commit_sha

    assert len(result.items) == 3

    code_change = next(item for item in result.items if item.feedback_id == "d" * 64)

    assert code_change.intent == "code_change"

    assert code_change.requires_code_change is True

    discussion = next(item for item in result.items if item.feedback_id == "e" * 64)

    assert discussion.intent == "discussion"

    assert discussion.requires_reply is True

    assert len(result.analysis_id) == 64

    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_analysis_is_deterministic():

    artifact = create_artifact()

    response = {
        "source_artifact_id": artifact.artifact_id,
        "items": [
            {
                "feedback_id": item.feedback_id,
                "intent": "non_actionable",
                "summary": "无需处理",
                "reasoning": "测试",
                "requires_code_change": False,
                "requires_reply": False,
                "target_hints": [],
            }
            for item in artifact.items
        ],
    }

    analyzer = ReviewFeedbackAnalyzer(model=FakeTextModel(response))

    first = await analyzer.analyze(artifact=artifact)

    second = await analyzer.analyze(artifact=artifact)

    assert first.analysis_id == second.analysis_id


@pytest.mark.asyncio
async def test_rejects_wrong_source_artifact():

    artifact = create_artifact()

    model = FakeTextModel(
        {
            "source_artifact_id": "9" * 64,
            "items": [],
        }
    )

    analyzer = ReviewFeedbackAnalyzer(model=model)

    with pytest.raises(
        ReviewFeedbackAnalysisError,
        match="source_artifact_id",
    ):
        await analyzer.analyze(artifact=artifact)


@pytest.mark.asyncio
async def test_rejects_missing_feedback():

    artifact = create_artifact()

    model = FakeTextModel(
        {
            "source_artifact_id": artifact.artifact_id,
            "items": [
                {
                    "feedback_id": "d" * 64,
                    "intent": "code_change",
                    "summary": "修改代码",
                    "reasoning": "Reviewer 要求修改",
                    "requires_code_change": True,
                    "requires_reply": False,
                    "target_hints": [],
                }
            ],
        }
    )

    analyzer = ReviewFeedbackAnalyzer(model=model)

    with pytest.raises(
        ReviewFeedbackAnalysisError,
        match="遗漏了 Feedback",
    ):
        await analyzer.analyze(artifact=artifact)


@pytest.mark.asyncio
async def test_rejects_unexpected_feedback():

    artifact = create_artifact()

    items = [
        {
            "feedback_id": item.feedback_id,
            "intent": "non_actionable",
            "summary": "无需处理",
            "reasoning": "测试",
            "requires_code_change": False,
            "requires_reply": False,
            "target_hints": [],
        }
        for item in artifact.items
    ]

    items.append(
        {
            "feedback_id": "9" * 64,
            "intent": "non_actionable",
            "summary": "不存在的反馈",
            "reasoning": "测试",
            "requires_code_change": False,
            "requires_reply": False,
            "target_hints": [],
        }
    )

    model = FakeTextModel(
        {
            "source_artifact_id": artifact.artifact_id,
            "items": items,
        }
    )

    analyzer = ReviewFeedbackAnalyzer(model=model)

    with pytest.raises(
        ReviewFeedbackAnalysisError,
        match="不存在的 Feedback",
    ):
        await analyzer.analyze(artifact=artifact)


@pytest.mark.asyncio
async def test_rejects_duplicate_feedback():

    artifact = create_artifact()

    duplicated = {
        "feedback_id": "d" * 64,
        "intent": "code_change",
        "summary": "修改",
        "reasoning": "测试",
        "requires_code_change": True,
        "requires_reply": False,
        "target_hints": [],
    }

    model = FakeTextModel(
        {
            "source_artifact_id": artifact.artifact_id,
            "items": [
                duplicated,
                duplicated,
                {
                    "feedback_id": "e" * 64,
                    "intent": "discussion",
                    "summary": "讨论",
                    "reasoning": "测试",
                    "requires_code_change": False,
                    "requires_reply": True,
                    "target_hints": [],
                },
                {
                    "feedback_id": "f" * 64,
                    "intent": "non_actionable",
                    "summary": "无需处理",
                    "reasoning": "测试",
                    "requires_code_change": False,
                    "requires_reply": False,
                    "target_hints": [],
                },
            ],
        }
    )

    analyzer = ReviewFeedbackAnalyzer(model=model)

    with pytest.raises(
        ReviewFeedbackAnalysisError,
        match="重复的 feedback_id",
    ):
        await analyzer.analyze(artifact=artifact)


@pytest.mark.asyncio
async def test_rejects_inconsistent_code_change():

    artifact = create_artifact()

    response_items = []

    for item in artifact.items:

        if item.feedback_id == "d" * 64:
            response_items.append(
                {
                    "feedback_id": item.feedback_id,
                    "intent": "code_change",
                    "summary": "修改代码",
                    "reasoning": "Reviewer 明确要求修改",
                    "requires_code_change": False,
                    "requires_reply": False,
                    "target_hints": [],
                }
            )

        else:
            response_items.append(
                {
                    "feedback_id": item.feedback_id,
                    "intent": "non_actionable",
                    "summary": "无需处理",
                    "reasoning": "测试",
                    "requires_code_change": False,
                    "requires_reply": False,
                    "target_hints": [],
                }
            )

    analyzer = ReviewFeedbackAnalyzer(
        model=FakeTextModel(
            {
                "source_artifact_id": artifact.artifact_id,
                "items": response_items,
            }
        )
    )

    with pytest.raises(
        ReviewFeedbackAnalysisError,
        match=("requires_code_change=true"),
    ):
        await analyzer.analyze(artifact=artifact)


@pytest.mark.asyncio
async def test_rejects_invalid_json():

    artifact = create_artifact()

    class InvalidModel:

        async def complete(
            self,
            *,
            system_message: str,
            user_message: str,
        ) -> str:

            return "not json"

    analyzer = ReviewFeedbackAnalyzer(model=InvalidModel())

    with pytest.raises(
        ReviewFeedbackAnalysisError,
        match="契约",
    ):
        await analyzer.analyze(artifact=artifact)


@pytest.mark.asyncio
async def test_store_round_trip(
    tmp_path: Path,
):

    repository = tmp_path / "repository"

    repository.mkdir()

    artifact = create_artifact()

    model = FakeTextModel(
        {
            "source_artifact_id": artifact.artifact_id,
            "items": [
                {
                    "feedback_id": item.feedback_id,
                    "intent": "non_actionable",
                    "summary": "无需处理",
                    "reasoning": "测试",
                    "requires_code_change": False,
                    "requires_reply": False,
                    "target_hints": [],
                }
                for item in artifact.items
            ],
        }
    )

    analyzer = ReviewFeedbackAnalyzer(model=model)

    analysis = await analyzer.analyze(artifact=artifact)

    store = ReviewFeedbackAnalysisStore(repository)

    store.save(analysis)

    loaded = store.load(analysis.analysis_id)

    assert loaded == analysis
