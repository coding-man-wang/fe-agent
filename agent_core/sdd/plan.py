import json

from agent_core.llm.model import ChatRequest, Message
from agent_core.sdd.model import (
    RequirementSpec,
    TechnicalSpec,
    ProjectContext,
    ImplementationPlan,
)


class ImplementationPlanGenerator:
    def __init__(self, llm):
        self.llm = llm

    async def generate(
        self,
        requirement: RequirementSpec,
        technical_spec: TechnicalSpec,
        project_context: ProjectContext,
    ) -> ImplementationPlan:
        schema = ImplementationPlan.model_json_schema()

        system_prompt = f"""
        你是一名资深软件研发负责人。
        
        你的任务是根据已经确认的：
        
        - RequirementSpec
        - TechnicalSpec
        - ProjectContext
        
        生成一个可以指导后续开发执行的 ImplementationPlan。
        
        ImplementationPlan 关注的是：
        
        “应该按照什么顺序实施技术方案？”
        
        而不是直接生成具体代码。
        
        请遵守以下规则：
        
        1. 所有实施步骤必须服务于 RequirementSpec，
           不得擅自扩大需求范围。
        
        2. 必须遵守 TechnicalSpec 中已经确定的技术方案，
           不要重新设计另一套架构。
        
        3. 必须遵守 ProjectContext 中的工程规则。
        
        4. 步骤顺序必须符合真实软件开发依赖关系。
        
        5. 如果某个未确认信息会阻止后续开发，
           应该放入 preconditions 或 open_questions，
           不要自行假设。
        
        6. 每个 ImplementationStep 应该代表一个有意义的
           开发阶段，不要细化到单个函数、单行代码或单个文件修改。
        
        7. actions 描述当前阶段需要完成的主要动作。
        
        8. verification 描述当前阶段完成后应该如何验证。
        
        9. testing_strategy 中的重要测试方案必须在计划中体现。
        
        10. TechnicalSpec 中的重要风险需要在 risks 中保留。
        
        11. completion_criteria 应该能够判断整个需求是否已经真正完成。
        
        12. 不要输出具体代码。
        
        13. 不要生成最终 Task 列表。
            Task 拆分将在下一阶段完成。
        
        14. steps.order 从 1 开始并按照执行顺序递增。
        
        15. 只返回 JSON，不返回 Markdown。
        
        输出必须满足以下 JSON Schema：
        
        {json.dumps(schema, ensure_ascii=False)}
        """

        input_data = {
            "requirement": (requirement.model_dump()),
            "technical_spec": (technical_spec.model_dump()),
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

        return ImplementationPlan.model_validate_json(response.content)
