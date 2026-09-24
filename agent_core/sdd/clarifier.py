import json

from agent_core.llm.model import ChatRequest, Message
from agent_core.sdd.model import (
    RequirementSpec,
    RequirementValidationResult,
    RequirementPatch,
)


class RequirementClarifier:
    def __init__(self, llm):
        self.llm = llm

    async def create_path(
        self,
        spec: RequirementSpec,
        validation: RequirementValidationResult,
        user_answer: str,
    ) -> RequirementPatch:
        schema = RequirementPatch.model_json_schema()

        system_prompt = f"""
        你是一名软件需求变更分析人员。
        
        你的任务不是重新生成完整 RequirementSpec，
        而是根据用户本次补充的信息，
        生成一个 RequirementPatch。
        
        请遵守以下规则：
        
        1. 只修改用户本次回答明确影响的内容。
        
        2. 不要重新生成或重复输出没有发生变化的需求内容。
        
        3. 用户没有明确提供的信息，不要自行补充。
        
        4. 对 list 类型字段：
           - 新增内容放入 add。
           - 已经不再成立的原内容放入 remove。
        
        5. remove 中的内容必须尽量与原 RequirementSpec
           中已有的字符串完全一致。
        
        6. 如果用户已经回答某个 open question，
           将该问题的原始文本放入 open_questions.remove。
        
        7. 如果用户回答产生了新的未决问题，
           放入 open_questions.add。
        
        8. 如果用户回答推翻了之前某项 assumption，
           将原 assumption 放入 assumptions.remove。
        
        9. 如果用户回答确认了新的技术或业务限制，
           可以加入 constraints.add。
        
        10. 不要扩大需求范围。
        
        11. 只返回 JSON，不返回 Markdown。
        
        输出必须满足以下 JSON Schema：
        
        {json.dumps(schema, ensure_ascii=False)}
        """

        input_data = {
            "current_requirement": spec.model_dump(),
            "validation_result": validation.model_dump(),
            "user_answer": user_answer,
        }

        request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content=system_prompt,
                ),
                Message(
                    role="user",
                    content=json.dumps(input_data, ensure_ascii=False, indent=2),
                ),
            ]
        )

        response = await self.llm.chat(request)

        return RequirementPatch.model_validate_json(response.content)
