import json

from pydantic import (
    ValidationError,
)

from agent_core.llm.base import LLM
from agent_core.llm.model import (
    ChatRequest,
    Message,
)
from agent_core.sdd.model import (
    ImplementationPlan,
    ProjectContext,
    RequirementSpec,
    TaskPlan,
    TechnicalSpec,
)
from agent_core.sdd.task_validation import (
    validate_task_plan,
)


class TaskPlanContractError(ValueError):
    pass


class TaskPlanner:

    def __init__(
        self,
        llm: LLM,
        *,
        max_validation_retries: int = 1,
    ):

        if max_validation_retries < 0:
            raise ValueError("max_validation_retries " "不能小于 0")

        self.llm = llm

        self.max_validation_retries = max_validation_retries

    async def generate(
        self,
        requirement: RequirementSpec,
        technical_spec: TechnicalSpec,
        implementation_plan: ImplementationPlan,
        project_context: ProjectContext,
    ) -> TaskPlan:

        schema = TaskPlan.model_json_schema()

        allowed_repository_hints = project_context.repository_hint_candidates()

        system_prompt = f"""
你是一名资深软件研发任务规划人员。

当前阶段是：

    Pre-Investigation Logical Planning

即：

    你还没有充分阅读 Repository，
    因此不能把仓库假设写成执行事实。

你的任务不是决定：

    “具体修改哪个文件”

而是决定：

    “为了满足 Requirement，
     必须解决哪些逻辑责任问题，
     需要调查哪些仓库事实，
     任务之间有什么依赖，
     如何验证完成。”

核心原则：

    Plan early, commit late.

可以早点规划，
但在获得 Repository Evidence 前，
不要承诺具体文件级执行范围。

输入包括：

- RequirementSpec
- TechnicalSpec
- ImplementationPlan
- ProjectContext
- allowed_repository_hints

硬性规则：

1. planning_stage 必须是：

       "logical"

2. Task 必须来自 Requirement /
   TechnicalSpec / ImplementationPlan，
   不得扩大需求范围。

3. 每个非 verification Task
   必须填写 responsibilities。

4. responsibilities 描述：

   - 行为责任
   - 数据责任
   - Runtime Path 责任
   - Contract 责任
   - 测试责任

   例如：

       “维护筛选状态”
       “传播请求参数”
       “保持 Mock/Remote 语义一致”

   不要写成：

       “修改 src/views/X.vue”

5. investigation Task 必须填写
   investigation_questions。

   问题应描述：

       “需要从 Repository
        确认什么事实？”

   例如：

       “当前筛选状态由哪一层维护？”
       “Remote Runtime Path
        如何传递 status？”
       “Mock 是否具有独立过滤职责？”

6. code_change Task
   必须通过 depends_on
   直接或间接依赖
   至少一个 investigation Task。

7. investigation 负责收集事实，
   不直接修改业务代码。

8. code_change 负责在
   Repository Evidence 已建立后
   表达需要完成的代码责任。

9. test 负责测试实现。

10. verification 负责最终质量门禁。

11. target_modules 是历史兼容字段。

    v1.3 新生成的 Logical Plan：

        target_modules 必须为 []

12. repository_hints 只是
    Search Hint，
    不是 Write Permission，
    更不是最终修改目标。

13. repository_hints
    只能逐字使用
    allowed_repository_hints
    中存在的值。

14. 不允许根据：

    TechnicalSpec.affected_modules
    ImplementationPlan.affected_modules

    自己推导新的目录或文件路径。

15. 如果没有可靠 Repository Hint，
    repository_hints 可以为空。

    不要为了填字段而猜路径。

16. ProjectContext.relevant_files
    若为空，
    不代表信息缺失，
    更不能自行补具体文件。

17. depends_on 只表达真实的
    逻辑/证据依赖。

18. 没有真实依赖的 Task
    不要强行串行。

19. source_step_order
    必须引用真实
    ImplementationPlan step.order。

20. acceptance_criteria_refs
    必须逐字引用
    RequirementSpec 中真实存在的
    acceptance_criteria。

21. expected_outputs
    描述逻辑结果或 Artifact，
    不要虚构文件。

22. verification 必须明确说明
    如何判断当前 Task 已完成。

23. ImplementationPlan 中的
    test / verification 内容
    必须映射到 TaskPlan。

24. 不生成代码。

25. 只输出 JSON 对象。

26. 不输出 Markdown。

27. 不输出 ```json code fence。

输出必须满足以下 JSON Schema：

{json.dumps(
    schema,
    ensure_ascii=False,
)}
""".strip()

        input_data = {
            "requirement": (requirement.model_dump()),
            "technical_spec": (technical_spec.model_dump()),
            "implementation_plan": (implementation_plan.model_dump()),
            "project_context": (project_context.model_dump()),
            "allowed_repository_hints": (allowed_repository_hints),
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

        raw_output = response.content or ""

        last_error: Exception | None = None

        for attempt in range(self.max_validation_retries + 1):

            try:

                plan = TaskPlan.model_validate_json(self._extract_json_text(raw_output))

                validation = validate_task_plan(
                    plan,
                    requirement,
                    implementation_plan,
                    project_context,
                )

                contract_errors = list(validation.errors)

                if plan.planning_stage != "logical":
                    contract_errors.insert(
                        0,
                        ("TaskPlanner v1.3 " "必须输出 " "planning_stage=logical"),
                    )

                if contract_errors:
                    raise (TaskPlanContractError("; ".join(contract_errors)))

                return plan

            except (
                ValidationError,
                json.JSONDecodeError,
                TaskPlanContractError,
            ) as exc:

                last_error = exc

                if attempt >= self.max_validation_retries:
                    break

                raw_output = await self._repair_output(
                    raw_output=(raw_output),
                    error=exc,
                    schema=schema,
                    allowed_repository_hints=(allowed_repository_hints),
                )

        raise TaskPlanContractError(
            "TaskPlan 在有限修复后"
            "仍不满足 v1.3 "
            "Logical Planning Contract: "
            f"{last_error}"
        ) from last_error

    async def _repair_output(
        self,
        *,
        raw_output: str,
        error: Exception,
        schema: dict,
        allowed_repository_hints: list[str],
    ) -> str:

        repair_prompt = f"""
上一轮 TaskPlan 输出没有通过
v1.3 Logical Planning Contract。

这是一次 bounded contract repair。

你的任务只是修复：

- JSON / Schema
- DAG Contract
- Logical Planning Contract

不要重新扩大 Requirement，
不要增加新的业务目标。

特别注意：

1. planning_stage="logical"

2. target_modules=[]

3. 非 verification Task
   必须有 responsibilities

4. investigation Task
   必须有 investigation_questions

5. code_change 必须直接或间接
   依赖 investigation

6. repository_hints
   只能使用下面这个列表中的值：

{json.dumps(
    allowed_repository_hints,
    ensure_ascii=False,
)}

7. 不得发明具体文件路径。

8. 保持原有 Requirement、
   Acceptance Criteria
   和 Task Intent。

9. 只输出 JSON。

10. 不输出 Markdown。

Schema:

{json.dumps(
    schema,
    ensure_ascii=False,
)}

Validation Error:

{str(error)}
""".strip()

        request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content=repair_prompt,
                ),
                Message(
                    role="user",
                    content=raw_output,
                ),
            ]
        )

        response = await self.llm.chat(request)

        return response.content or ""

    @staticmethod
    def _extract_json_text(
        raw_output: str,
    ) -> str:

        text = raw_output.strip()

        if text.startswith("```") and text.endswith("```"):

            lines = text.splitlines()

            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        return text
