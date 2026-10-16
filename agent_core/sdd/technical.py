import json

from agent_core.llm.base import LLM
from agent_core.llm.model import (
    ChatRequest,
    Message,
)
from agent_core.sdd.model import (
    ProjectContext,
    RequirementSpec,
    TechnicalSpec,
)


class TechnicalSpecGenerator:

    def __init__(
        self,
        llm: LLM,
    ):
        self.llm = llm

    async def generate(
        self,
        requirement: RequirementSpec,
        project_context: ProjectContext,
    ) -> TechnicalSpec:

        schema = TechnicalSpec.model_json_schema()

        system_prompt = f"""
你是一名资深前端软件架构师。

你的任务是：

根据已经确认的 RequirementSpec
和当前项目的 ProjectContext，
生成可以指导后续开发的 TechnicalSpec。

请遵守以下规则：

1. TechnicalSpec 必须服务于 RequirementSpec，
   不要擅自增加需求范围。

2. 优先复用 ProjectContext 中已经存在的
   技术栈、组件和工程基础设施。

3. 不要为了实现简单需求擅自引入新的
   框架、库、中间件或复杂架构。

4. 如果确实需要引入新的依赖，
   必须在 technical_decisions 中说明原因。

5. RequirementSpec 中的 constraints 必须遵守。

6. RequirementSpec 中的 non_goals 不得被方案重新引入。

7. ProjectContext 中的 engineering_rules 必须遵守。

8. 不确定的技术事实不要编造。
   合理但未确认的前提放入 assumptions。

9. 会显著影响后续实现的问题放入 open_questions。

10. technical_decisions 必须包含：
    - topic
    - decision
    - rationale

11. testing_strategy 必须对应本次需求的
    Acceptance Criteria 和主要风险。

12. risks 只记录真实可能影响实施、
    稳定性或正确性的风险。

13. 不要输出具体代码。

14. 当前阶段只做技术设计，
    不要直接生成 Implementation Task。

15. 只返回 JSON，不返回 Markdown。

输出必须符合以下 JSON Schema：

{json.dumps(schema, ensure_ascii=False)}
"""

        input_data = {
            "requirement": (requirement.model_dump()),
            "project_context": (project_context.model_dump()),
        }

        request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content=system_prompt,
                ),
                Message(
                    role="user",
                    content=json.dumps(
                        input_data,
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
            ]
        )

        response = await self.llm.chat(request)

        return TechnicalSpec.model_validate_json(response.content)
