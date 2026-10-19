import json

from pydantic import ValidationError

from agent_core.change_surface.errors import InvestigationArtifactBuildError
from agent_core.change_surface.model import (
    InvestigationArtifact,
    ObservedRepositoryFile,
)
from agent_core.context.model import TaskContextBundle
from agent_core.llm.base import LLM
from agent_core.llm.model import ChatRequest, Message


class InvestigationArtifactBuilder:
    """把 Investigation 的真实代码上下文转成结构化 Repository Facts。"""

    def __init__(self, *, llm: LLM) -> None:
        self.llm = llm

    async def build(self, *, bundle: TaskContextBundle) -> InvestigationArtifact:
        self._validate_input(bundle=bundle)
        schema = InvestigationArtifact.model_json_schema()
        request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content=self._build_system_prompt(schema=schema),
                ),
                Message(
                    role="user",
                    content=json.dumps(
                        self._build_payload(bundle=bundle),
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
            ]
        )
        response = await self.llm.chat(request)
        content = response.content.strip()
        if not content:
            raise InvestigationArtifactBuildError("LLM 返回空 InvestigationArtifact")

        try:
            artifact = InvestigationArtifact.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            raise InvestigationArtifactBuildError(
                "LLM 返回的 InvestigationArtifact 不满足 Schema"
            ) from exc

        artifact = self._canonicalize_paths(artifact=artifact, bundle=bundle)
        self._validate_output(artifact=artifact, bundle=bundle)
        return artifact

    @staticmethod
    def _validate_input(*, bundle: TaskContextBundle) -> None:
        if bundle.task.task_type != "investigation":
            raise InvestigationArtifactBuildError(
                "InvestigationArtifactBuilder 只能处理 investigation Task"
            )
        if not bundle.code_context:
            raise InvestigationArtifactBuildError(
                "Investigation Task 没有 Repository Code Context"
            )

    @staticmethod
    def _build_system_prompt(*, schema: dict) -> str:
        return f"""
你是一名资深软件仓库分析工程师。
你的任务不是修改代码，而是基于提供的 Repository Code Context 生成 InvestigationArtifact。

规则：
1. observed_files.relative_path 只能来自输入 code_context。
2. evidence 必须是代码事实，不能写“排名高”“很相关”等检索结论。
3. candidate_target 只是候选修改位置，不是最终修改决定。
4. dependency 表示需要理解但原则上不主动修改。
5. reference 表示类似实现或邻接实现。
6. test 表示测试文件。
7. unknown 表示证据不足。
8. Requirement/TechnicalSpec 明确声明“不修改”的模块，如仍需理解，应优先标 dependency。
9. findings 只能记录已从代码确认的事实。
10. 未提供文件才能回答的问题进入 unresolved_questions，不要猜。
11. 不得声称已修改代码或已运行测试。
12. 不得读取或使用 Benchmark Oracle / expected_files。
13. 只返回 JSON，不返回 Markdown。

JSON Schema：
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()

    @staticmethod
    def _build_payload(*, bundle: TaskContextBundle) -> dict:
        return {
            "context_type": "repository_investigation",
            "task": bundle.task.model_dump(mode="json"),
            "requirement": {
                "title": bundle.requirement.title,
                "summary": bundle.requirement.summary,
                "goals": list(bundle.requirement.goals),
                "non_goals": list(bundle.requirement.non_goals),
                "acceptance_criteria": list(bundle.requirement.acceptance_criteria),
                "constraints": list(bundle.requirement.constraints),
            },
            "technical_spec": {
                "summary": bundle.technical_spec.summary,
                "solution_approach": list(bundle.technical_spec.solution_approach),
                "affected_modules": list(bundle.technical_spec.affected_modules),
                "reuse_plan": list(bundle.technical_spec.reuse_plan),
                "data_flow": list(bundle.technical_spec.data_flow),
                "technical_decisions": [
                    item.model_dump(mode="json")
                    for item in bundle.technical_spec.technical_decisions
                ],
                "open_questions": list(bundle.technical_spec.open_questions),
            },
            "repository": bundle.repository.model_dump(mode="json"),
            "engineering_rules": list(bundle.engineering_rules),
            "code_context": [
                {
                    "relative_path": item.relative_path,
                    "category": item.category,
                    "language": item.language,
                    "content_mode": item.content_mode,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "relevance_reasons": list(item.relevance_reasons),
                    "content": item.content,
                }
                for item in bundle.code_context
            ],
            "context_budget": bundle.budget.model_dump(mode="json"),
        }

    @staticmethod
    def _normalize_path(path: str) -> str:
        normalized = path.replace("\\", "/").strip()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    def _canonicalize_paths(
        self,
        *,
        artifact: InvestigationArtifact,
        bundle: TaskContextBundle,
    ) -> InvestigationArtifact:
        canonical_paths = {
            self._normalize_path(item.relative_path).casefold(): item.relative_path
            for item in bundle.code_context
        }
        normalized_files: list[ObservedRepositoryFile] = []
        for observed in artifact.observed_files:
            normalized = self._normalize_path(observed.relative_path)
            canonical = canonical_paths.get(normalized.casefold(), normalized)
            normalized_files.append(
                observed.model_copy(
                    deep=True,
                    update={"relative_path": canonical},
                )
            )
        return artifact.model_copy(
            deep=True,
            update={"observed_files": normalized_files},
        )

    def _validate_output(
        self,
        *,
        artifact: InvestigationArtifact,
        bundle: TaskContextBundle,
    ) -> None:
        if artifact.task_id != bundle.task.task_id:
            raise InvestigationArtifactBuildError(
                f"Artifact task_id 不一致: expected={bundle.task.task_id}, "
                f"actual={artifact.task_id}"
            )

        allowed_paths = {
            self._normalize_path(item.relative_path).casefold()
            for item in bundle.code_context
        }
        invalid = [
            item.relative_path
            for item in artifact.observed_files
            if self._normalize_path(item.relative_path).casefold() not in allowed_paths
        ]
        if invalid:
            raise InvestigationArtifactBuildError(
                "InvestigationArtifact 引用了未提供的文件: " + ", ".join(invalid)
            )
