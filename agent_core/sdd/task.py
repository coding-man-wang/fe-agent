import json

from agent_core.llm.base import LLM
from agent_core.llm.model import Message, ChatRequest
from agent_core.sdd.model import (
    RequirementSpec,
    TaskPlan,
    TechnicalSpec,
    ImplementationPlan,
    ProjectContext,
)


class TaskPlanner:
    def __init__(self, llm: LLM):
        self.llm = llm

    async def generate(
        self,
        requirement: RequirementSpec,
        technical_spec: TechnicalSpec,
        implementation_plan: ImplementationPlan,
        project_context: ProjectContext,
    ) -> TaskPlan:
        schema = TaskPlan.model_json_schema()

        system_prompt = f"""
    你是一名资深软件研发任务规划人员。

    你的任务是把已经确定的 ImplementationPlan
    转换成可以由后续 Agent Workflow 执行的 TaskPlan。

    输入包括：

    - RequirementSpec
    - TechnicalSpec
    - ImplementationPlan
    - ProjectContext

    请遵守以下规则：

    1. Task 必须来自 ImplementationPlan，
       不得自行增加需求范围。

    2. 每个 Task 应该有一个明确的主要目标，
       避免一个 Task 同时完成多个无关职责。

    3. Task 粒度应适合一次独立 Agent 执行，
       但不要细化到单行代码修改。

    4. 每个 Task 必须拥有唯一 task_id。

    5. task_id 使用：
       TASK-001
       TASK-002
       TASK-003
       这种格式。

    6. depends_on 中只能引用当前 TaskPlan
       中真实存在的 task_id。

    7. 不允许 Task 依赖自己。

    8. 不允许产生循环依赖。

    9. 如果某个 Task 必须等另一个 Task 完成，
       应明确写入 depends_on。

    10. 没有真实依赖关系的 Task
        不要为了保持顺序强行建立依赖。

    11. task_type 只能是：

        investigation
        code_change
        test
        verification

    12. investigation 用于确认技术事实或依赖，
        不直接修改业务代码。

    13. code_change 用于真实代码变更。

    14. test 用于创建或修改测试。

    15. verification 用于执行最终验证。

    16. expected_outputs 必须明确描述
        当前任务产生什么结果。

    17. verification 必须说明
        如何判断当前任务已经成功。

    18. acceptance_criteria_refs 必须直接引用
        RequirementSpec 中真实存在的
        acceptance_criteria 原文。

    19. 不要伪造 ProjectContext 中不存在的
        精确文件路径。

    20. 如果只知道模块，不知道具体文件，
        target_modules 应记录模块名称，
        不要自行猜测文件路径。

    21. source_step_order 必须对应
        ImplementationPlan 中真实的 step.order。

    22. ImplementationPlan 中的测试和验证内容
        必须反映到 TaskPlan。

    23. 不生成代码。

    24. 只返回 JSON，不返回 Markdown。

    输出必须满足以下 JSON Schema：

    {json.dumps(schema, ensure_ascii=False)}
    """

        input_data = {
            "requirement": requirement.model_dump(),
            "technical_spec": technical_spec.model_dump(),
            "implementation_plan": (implementation_plan.model_dump()),
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

        return TaskPlan.model_validate_json(response.content)
