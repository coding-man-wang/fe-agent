import json

from agent_core.llm.model import ChatRequest, Message
from agent_core.sdd.model import RequirementSpec, RequirementValidationResult


class RequireValidator:
    def __init__(self, llm):
        self.llm = llm

    async def validate(self, spec: RequirementSpec) -> RequirementValidationResult:
        schema = RequirementValidationResult.model_json_schema()

        system_prompt = f"""
        你是一名资深软件需求评审人员。

        你的任务不是重新生成需求，
        而是检查已有 RequirementSpec 是否已经足够清晰，
        可以进入 Technical Design 阶段。

        请重点检查：

        1. 是否缺少会显著影响技术实现的信息。
        2. 是否存在有歧义的描述。
        3. 不同需求之间是否存在冲突。
        4. Acceptance Criteria 是否可以被验证或测试。
        5. 是否存在明显的范围不确定问题。
        6. Assumptions 是否包含尚未确认、但会显著影响技术方案的假设。
        7. Open Questions 中是否存在必须先回答的问题。

        判断原则：

        - 只有会影响架构、实现方案、数据模型、接口设计或验收结果的问题，
          才应该视为 blocking。

        - 不影响主要技术设计的小问题可以标记为 warning。

        - 不要自行回答未知问题。

        - 如果存在 blocking 问题，
          is_ready 必须为 false。

        - 如果 is_ready 为 false，
          clarification_questions 必须给出需要向用户确认的问题。

        - clarification_questions 应该具体、简洁、可直接回答。

        - 不要扩大原始需求范围。

        只返回 JSON，不返回 Markdown。

        输出必须满足以下 JSON Schema：

        {json.dumps(schema, ensure_ascii=False)}
        """

        user_content = spec.model_dump_json()

        request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content=system_prompt,
                ),
                Message(
                    role="user",
                    content=user_content,
                )
            ]
        )

        response = await self.llm.chat(request)

        return RequirementValidationResult.model_validate_json(response.content)
