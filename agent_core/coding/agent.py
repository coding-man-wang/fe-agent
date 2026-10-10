from typing import Protocol

from pydantic import ValidationError

from agent_core.coding.model import (
    ReadOnlyCodingResult,
)
from agent_core.context.model import (
    TaskContextBundle,
    RenderedContext,
)
from agent_core.context.renderer import (
    ContextRenderer,
)


class TextModel(Protocol):

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str: ...


class CodingOutputError(ValueError):
    """模型输出不符合 Coding Proposal 契约。"""


class ReadOnlyCodingAgent:
    OUTPUT_INSTRUCTION = """
请只返回一个合法 JSON 对象，不要使用 Markdown 代码围栏。

JSON 必须符合以下结构：

{
  "task_id": "当前任务 ID",
  "summary": "本次代码分析摘要",
  "findings": [
    {
      "relative_path": "文件相对路径",
      "evidence": "提供的代码中实际存在的原文片段",
      "explanation": "基于证据的分析结论"
    }
  ],
  "proposed_changes": [
    {
      "relative_path": "需要修改的已有文件路径",
      "objective": "修改目标",
      "proposed_approach": "建议修改方式"
    }
  ],
  "verification_plan": [
    "建议执行的验证步骤"
  ],
  "missing_information": [
    "仍然缺少的信息"
  ],
  "execution_status": "not_executed"
}

要求：
1. 只能引用提供的 Code Context 中出现的文件路径。
2. findings.evidence 必须是相应代码内容的原文片段。
3. 如果代码不足以支持结论，不要编造证据；
   应放入 missing_information。
4. 只提出修改方案，不声称已经修改或测试。
5. 不要输出工具调用或可执行动作指令。
""".strip()

    def __init__(
        self,
        *,
        model: TextModel,
        renderer: ContextRenderer | None = None,
    ) -> None:

        self.model = model
        self.renderer = renderer or ContextRenderer()

    async def analyze(
        self,
        bundle: TaskContextBundle,
    ) -> ReadOnlyCodingResult:

        messages = self.prepare_messages(bundle)

        return await self.analyze_prepared(
            bundle=bundle,
            messages=messages,
        )

    def _validate_result(
        self,
        *,
        bundle: TaskContextBundle,
        result: ReadOnlyCodingResult,
    ) -> None:

        if result.task_id != bundle.task.task_id:
            raise CodingOutputError("模型返回的 task_id 与当前任务不一致")

        code_map = {item.relative_path: item.content for item in bundle.code_context}

        for finding in result.findings:

            content = code_map.get(finding.relative_path)

            if content is None:
                raise CodingOutputError(
                    "Finding 引用了未提供的文件: " f"{finding.relative_path}"
                )

            if finding.evidence not in content:
                raise CodingOutputError(
                    "Finding 的 evidence "
                    "不存在于对应代码上下文: "
                    f"{finding.relative_path}"
                )

        for change in result.proposed_changes:

            if change.relative_path not in code_map:
                raise CodingOutputError(
                    "修改提案引用了未提供的文件: " f"{change.relative_path}"
                )

    def prepare_messages(
        self,
        bundle: TaskContextBundle,
    ) -> RenderedContext:

        rendered = self.renderer.render(bundle)

        return RenderedContext(
            system_message=(rendered.system_message + "\n\n" + self.OUTPUT_INSTRUCTION),
            user_message=rendered.user_message,
        )

    async def analyze_prepared(
        self,
        *,
        bundle: TaskContextBundle,
        messages: RenderedContext,
    ) -> ReadOnlyCodingResult:

        if not bundle.code_context:
            raise CodingOutputError("当前 Task 没有可分析的代码上下文")

        raw_output = await self.model.complete(
            system_message=messages.system_message,
            user_message=messages.user_message,
        )

        try:
            result = ReadOnlyCodingResult.model_validate_json(raw_output)

        except ValidationError as exc:
            raise CodingOutputError("模型输出不符合 ReadOnlyCodingResult 契约") from exc

        self._validate_result(
            bundle=bundle,
            result=result,
        )

        return result
