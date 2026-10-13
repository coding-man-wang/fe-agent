import hashlib
import json

from agent_core.git.follow_up_task_model import (
    FollowUpTaskArtifact,
    FollowUpTaskLink,
)
from agent_core.git.review_feedback_analysis_model import (
    ReviewFeedbackAnalysisArtifact,
    ReviewFeedbackAnalysisItem,
)
from agent_core.git.review_feedback_model import (
    ReviewFeedbackArtifact,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


class FollowUpTaskError(ValueError):
    pass


class FollowUpTaskGenerator:

    def generate(
        self,
        *,
        parent_task: TaskDefinition,
        feedback: ReviewFeedbackArtifact,
        analysis: ReviewFeedbackAnalysisArtifact,
    ) -> FollowUpTaskArtifact:

        self._validate_sources(
            parent_task=parent_task,
            feedback=feedback,
            analysis=analysis,
        )

        feedback_map = {item.feedback_id: item for item in feedback.items}

        code_change_items = [
            item for item in analysis.items if item.intent == "code_change"
        ]

        code_change_items.sort(key=lambda item: (item.feedback_id))

        tasks: list[TaskDefinition] = []

        links: list[FollowUpTaskLink] = []

        task_ids: set[str] = set()

        for item in code_change_items:

            source_feedback = feedback_map.get(item.feedback_id)

            if source_feedback is None:
                raise FollowUpTaskError(
                    "Analysis 引用了不存在的 " "Review Feedback: " f"{item.feedback_id}"
                )

            task = self._build_task(
                parent_task=parent_task,
                analysis=analysis,
                item=item,
            )

            if task.task_id in task_ids:
                raise FollowUpTaskError("生成了重复的 " f"task_id: {task.task_id}")

            task_ids.add(task.task_id)

            tasks.append(task)

            links.append(
                FollowUpTaskLink(
                    feedback_id=(item.feedback_id),
                    follow_up_task_id=(task.task_id),
                )
            )

        artifact_id = self._calculate_artifact_id(
            parent_task=parent_task,
            feedback=feedback,
            analysis=analysis,
            tasks=tasks,
            links=links,
        )

        return FollowUpTaskArtifact(
            artifact_id=artifact_id,
            source_analysis_id=(analysis.analysis_id),
            source_feedback_artifact_id=(feedback.artifact_id),
            parent_task_id=(parent_task.task_id),
            commit_sha=(analysis.commit_sha),
            tasks=tasks,
            links=links,
        )

    def _validate_sources(
        self,
        *,
        parent_task: TaskDefinition,
        feedback: ReviewFeedbackArtifact,
        analysis: ReviewFeedbackAnalysisArtifact,
    ) -> None:

        if analysis.source_artifact_id != feedback.artifact_id:
            raise FollowUpTaskError("Analysis 与 Review Feedback " "Artifact 不匹配")

        if analysis.task_id != parent_task.task_id:
            raise FollowUpTaskError("Analysis task_id " "与 parent_task 不一致")

        if feedback.task_id != parent_task.task_id:
            raise FollowUpTaskError("Review Feedback task_id " "与 parent_task 不一致")

        if analysis.proposal_id != feedback.proposal_id:
            raise FollowUpTaskError("Analysis proposal_id " "与 Review Feedback 不一致")

        if analysis.commit_sha != feedback.commit_sha:
            raise FollowUpTaskError("Analysis commit_sha " "与 Review Feedback 不一致")

        feedback_ids = {item.feedback_id for item in feedback.items}

        analysis_ids = {item.feedback_id for item in analysis.items}

        if feedback_ids != analysis_ids:

            missing = sorted(feedback_ids - analysis_ids)

            unexpected = sorted(analysis_ids - feedback_ids)

            raise FollowUpTaskError(
                "Feedback 与 Analysis "
                "范围不一致: "
                f"missing={missing}, "
                f"unexpected={unexpected}"
            )

    def _build_task(
        self,
        *,
        parent_task: TaskDefinition,
        analysis: ReviewFeedbackAnalysisArtifact,
        item: ReviewFeedbackAnalysisItem,
    ) -> TaskDefinition:

        if not item.requires_code_change:
            raise FollowUpTaskError("code_change Analysis " "必须要求代码修改")

        task_id = self._build_task_id(
            parent_task_id=(parent_task.task_id),
            analysis_id=(analysis.analysis_id),
            feedback_id=(item.feedback_id),
        )

        inputs = [
            ("Review Feedback Analysis: " f"{item.summary}"),
            ("Review Feedback Reasoning: " f"{item.reasoning}"),
            ("Source Review Feedback ID: " f"{item.feedback_id}"),
            ("Source Analysis ID: " f"{analysis.analysis_id}"),
        ]

        return TaskDefinition(
            task_id=task_id,
            task_type="code_change",
            title=("处理 Code Review 反馈：" f"{item.summary}"),
            objective=(
                "根据已确认的 Code Review " "反馈完成必要代码调整：" f"{item.summary}"
            ),
            source_step_order=(parent_task.source_step_order),
            depends_on=[],
            inputs=inputs,
            target_modules=(self._normalize_hints(item.target_hints)),
            expected_outputs=[
                ("完成 Review Feedback " f"{item.feedback_id} " "要求的代码调整"),
                ("保持修改范围与原始需求" "及 Review Feedback 一致"),
            ],
            verification=[
                ("验证 Review Feedback " f"{item.feedback_id} " "已被实际代码变更覆盖"),
                ("执行与修改范围相关的" "现有测试或验证步骤"),
            ],
            acceptance_criteria_refs=list(parent_task.acceptance_criteria_refs),
        )

    def _normalize_hints(
        self,
        hints: list[str],
    ) -> list[str]:

        seen: set[str] = set()

        result: list[str] = []

        for raw in hints:

            value = raw.strip()

            if not value:
                continue

            key = value.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(value)

        return result

    def _build_task_id(
        self,
        *,
        parent_task_id: str,
        analysis_id: str,
        feedback_id: str,
    ) -> str:

        raw = f"{parent_task_id}:" f"{analysis_id}:" f"{feedback_id}"

        suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]

        return f"{parent_task_id}" f"-REV-{suffix}"

    def _calculate_artifact_id(
        self,
        *,
        parent_task: TaskDefinition,
        feedback: ReviewFeedbackArtifact,
        analysis: ReviewFeedbackAnalysisArtifact,
        tasks: list[TaskDefinition],
        links: list[FollowUpTaskLink],
    ) -> str:

        payload = {
            "parent_task_id": (parent_task.task_id),
            "source_feedback_artifact_id": (feedback.artifact_id),
            "source_analysis_id": (analysis.analysis_id),
            "commit_sha": (analysis.commit_sha),
            "tasks": [task.model_dump(mode="json") for task in tasks],
            "links": [link.model_dump(mode="json") for link in links],
        }

        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()
