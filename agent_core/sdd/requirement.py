import json

from agent_core.llm.model import ChatRequest, Message
from agent_core.sdd.model import RequirementSpec


class RequirementAnalyzer:
    def __init__(self, llm):
        self.llm = llm

    async def analyze(self, requirement: str) -> RequirementSpec:
        schema = RequirementSpec.model_json_schema()

        system_prompt = f"""你是一名资深软件需求分析师。
你的任务是把用户输入的自然语言需求，
整理成结构化的软件需求规格。

请遵守以下原则：

1. 不要擅自扩大用户需求范围。
2. 用户没有明确说明的信息，不要编造。
3. 不确定的信息放入 open_questions。
4. 合理但尚未确认的前提放入 assumptions。
5. acceptance_criteria 必须尽量可验证。
6. constraints 记录技术、业务或工程约束。
7. 只返回 JSON，不要返回 Markdown。
8. 返回结果必须满足下面的 JSON Schema。

JSON Schema：

{json.dumps(schema, ensure_ascii=False)}
"""
        request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content=system_prompt,
                ),
                Message(
                    role="user",
                    content=requirement
                )
            ]
        )

        response = await self.llm.chat(request)
        
        return RequirementSpec.model_validate_json(response.content)
