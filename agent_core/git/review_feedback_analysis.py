import hashlib
import json
from typing import Protocol

from pydantic import ValidationError

from agent_core.git.review_feedback_analysis_model import (
    ReviewFeedbackAnalysisArtifact,
    ReviewFeedbackAnalysisOutput,
)
from agent_core.git.review_feedback_model import (
    ReviewFeedbackArtifact,
)


class TextModel(Protocol):

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str: ...


class ReviewFeedbackAnalysisError(ValueError):
    pass


class ReviewFeedbackAnalyzer:

    SYSTEM_INSTRUCTION = """
你是一名代码审查反馈分析 Agent。

你的职责仅限于分析 Pull Request Reviewer 的反馈，
不得修改代码，不得声称已经执行任何操作。

你需要判断每一条反馈属于哪一种 intent：

1. code_change
   Reviewer 明确要求修改代码、测试、配置或实现。

2. discussion
   Reviewer 在询问设计原因、讨论方案或表达需要回复的问题，
   当前无法直接判断必须修改代码。

3. clarification_needed
   Reviewer 明显希望进行处理，
   但反馈信息不足，无法安全确定具体修改方式。

4. non_actionable
   无需进一步行动，例如 LGTM、感谢、纯状态性评论。

输出要求：

- 每个输入 feedback_id 必须且只能出现一次。
- 不得新增不存在的 feedback_id。
- target_hints 只能表示可能相关的文件、模块、符号或概念，
  它只是后续 Retrieval Hint，不是已经验证的修改范围。
- 不得生成具体代码。
- 不得声称已经修改、测试、提交或回复。
- 只返回合法 JSON，不要使用 Markdown 代码围栏。

JSON 格式：

{
  "source_artifact_id": "...",
  "items": [
    {
      "feedback_id": "...",
      "intent": "code_change",
      "summary": "...",
      "reasoning": "...",
      "requires_code_change": true,
      "requires_reply": false,
      "target_hints": [
        "src/example.py",
        "SomeSymbol"
      ]
    }
  ]
}
""".strip()

    def __init__(
        self,
        *,
        model: TextModel,
    ) -> None:

        self.model = model

    async def analyze(
        self,
        *,
        artifact: ReviewFeedbackArtifact,
    ) -> ReviewFeedbackAnalysisArtifact:

        user_message = self._build_user_message(artifact)

        raw_output = await self.model.complete(
            system_message=(self.SYSTEM_INSTRUCTION),
            user_message=(user_message),
        )

        try:

            output = ReviewFeedbackAnalysisOutput.model_validate_json(raw_output)

        except ValidationError as exc:

            raise (
                ReviewFeedbackAnalysisError(
                    "模型输出不符合 " "ReviewFeedbackAnalysisOutput " "契约"
                )
            ) from exc

        self._validate_output(
            artifact=artifact,
            output=output,
        )

        analysis_id = self._calculate_analysis_id(
            artifact=artifact,
            output=output,
        )

        return ReviewFeedbackAnalysisArtifact(
            analysis_id=analysis_id,
            source_artifact_id=(artifact.artifact_id),
            task_id=(artifact.task_id),
            proposal_id=(artifact.proposal_id),
            commit_sha=(artifact.commit_sha),
            items=output.items,
        )

    def _build_user_message(
        self,
        artifact: ReviewFeedbackArtifact,
    ) -> str:

        payload = {
            "source_artifact_id": (artifact.artifact_id),
            "pull_request": {
                "provider": (artifact.pull_request_provider),
                "repository": (artifact.repository),
                "number": (artifact.pull_request_number),
                "task_id": (artifact.task_id),
                "proposal_id": (artifact.proposal_id),
                "commit_sha": (artifact.commit_sha),
            },
            "feedback": [
                {
                    "feedback_id": (item.feedback_id),
                    "kind": (item.kind),
                    "reviewer": (item.reviewer),
                    "body": (item.body),
                    "relative_path": (item.relative_path),
                    "line": (item.line),
                    "comment_commit_sha": (item.commit_sha),
                }
                for item in artifact.items
            ],
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )

    def _validate_output(
        self,
        *,
        artifact: ReviewFeedbackArtifact,
        output: ReviewFeedbackAnalysisOutput,
    ) -> None:

        if output.source_artifact_id != artifact.artifact_id:
            raise (
                ReviewFeedbackAnalysisError(
                    "模型返回的 " "source_artifact_id " "与当前 Artifact 不一致"
                )
            )

        expected_ids = {item.feedback_id for item in artifact.items}

        returned_ids = [item.feedback_id for item in output.items]

        if len(returned_ids) != len(set(returned_ids)):
            raise (ReviewFeedbackAnalysisError("模型返回了重复的 " "feedback_id"))

        returned_id_set = set(returned_ids)

        missing = sorted(expected_ids - returned_id_set)

        unexpected = sorted(returned_id_set - expected_ids)

        if missing:
            raise (ReviewFeedbackAnalysisError("模型遗漏了 Feedback: " f"{missing}"))

        if unexpected:
            raise (
                ReviewFeedbackAnalysisError(
                    "模型返回了不存在的 " "Feedback: " f"{unexpected}"
                )
            )

        for item in output.items:

            self._validate_item(item)

    def _validate_item(
        self,
        item,
    ) -> None:

        if item.intent == "code_change" and not item.requires_code_change:
            raise (
                ReviewFeedbackAnalysisError(
                    "code_change " "必须设置 " "requires_code_change=true"
                )
            )

        if item.intent == "non_actionable":

            if item.requires_code_change:
                raise (
                    ReviewFeedbackAnalysisError("non_actionable " "不能要求代码修改")
                )

            if item.requires_reply:
                raise (ReviewFeedbackAnalysisError("non_actionable " "不能要求回复"))

        if (
            item.intent
            in {
                "discussion",
                "clarification_needed",
            }
            and not item.requires_reply
        ):
            raise (
                ReviewFeedbackAnalysisError(
                    f"{item.intent} " "必须设置 " "requires_reply=true"
                )
            )

    def _calculate_analysis_id(
        self,
        *,
        artifact: ReviewFeedbackArtifact,
        output: ReviewFeedbackAnalysisOutput,
    ) -> str:

        payload = {
            "source_artifact_id": (artifact.artifact_id),
            "task_id": (artifact.task_id),
            "proposal_id": (artifact.proposal_id),
            "commit_sha": (artifact.commit_sha),
            "analysis": (output.model_dump(mode="json")),
        }

        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()
