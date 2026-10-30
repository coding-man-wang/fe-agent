import json

from pydantic import ValidationError

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
        *,
        max_validation_retries: int = 1,
    ):
        if max_validation_retries < 0:
            raise ValueError("max_validation_retries 不能小于 0")

        self.llm = llm
        self.max_validation_retries = max_validation_retries

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

        当前阶段仍处于：

            Pre-Investigation Technical Design

        即：

            系统还没有通过 Repository Investigation
            充分读取和确认仓库代码事实。

        因此 TechnicalSpec 的职责是：

            描述系统应该如何实现、
            哪些逻辑责任可能受到影响、
            数据应该如何流转、
            存在哪些技术约束和风险。

        TechnicalSpec 不负责：

            确定最终修改文件，
            确定最终 Change Surface，
            或授予任何 Write Permission。

        请遵守以下规则：

        1. TechnicalSpec 必须服务于 RequirementSpec，
           不要擅自增加需求范围。

        2. 优先复用 ProjectContext 中已经明确存在的
           技术栈、组件和工程基础设施。

        3. 不要为了实现简单需求擅自引入新的
           框架、库、中间件或复杂架构。

        4. 如果确实需要引入新的依赖，
           必须在 technical_decisions 中说明原因。

        5. RequirementSpec 中的 constraints
           必须遵守。

        6. RequirementSpec 中的 non_goals
           不得被方案重新引入。

        7. ProjectContext 中的 engineering_rules
           必须遵守。

        8. 不确定的技术事实不要编造。

           合理但尚未被 Repository Evidence
           确认的前提，应放入 assumptions。

        9. 会显著影响后续实现、
           且必须通过读取 Repository
           才能确认的问题，
           应放入 open_questions。

        10. affected_modules 表示：

                “预计可能受到影响的逻辑模块、
                 技术层或责任区域”

            它不是：

                最终文件级修改计划，
                Change Surface，
                或 Write Permission。

        11. affected_modules 只能基于
            RequirementSpec 和 ProjectContext
            当前已经提供的信息填写。

            如果 ProjectContext 只提供了：

                src/api
                src/views
                src/stores

            这样的目录级事实，

            可以使用这些已经明确存在的
            模块或目录信息。

        12. 不允许根据目录名、
            需求名称或常见项目结构，
            自行推导 ProjectContext
            没有明确提供的具体文件。

            例如：

                已知 src/api

            不等于可以推导：

                src/api/orders.ts

            已知 src/views

            不等于可以推导：

                src/views/OrdersView.vue

        13. ProjectContext.relevant_files
            如果明确提供了具体文件，
            可以把它们作为已知 Repository Fact。

            如果 relevant_files 为空，
            不要自行补充具体文件。

        14. solution_approach 应描述：

            - 需要实现什么行为；
            - 哪些逻辑责任需要协作；
            - 数据如何流转；
            - 哪些 Contract 需要保持；
            - 哪些技术风险需要验证。

            不要把 solution_approach
            写成未经证据确认的文件修改清单。

        15. data_flow 描述的是：

                数据、状态、参数、事件
                如何在不同责任层之间流转。

            如果具体 Repository Runtime Path
            尚未被确认，
            应使用责任层级描述，
            不要虚构具体函数或文件。

        16. technical_decisions 必须包含：

            - topic
            - decision
            - rationale

        17. technical_decisions 中：

            已由 Requirement / ProjectContext
            明确支持的内容可以作为 decision；

            仍依赖 Repository 实现事实的内容，
            不要伪装成已经确认的结论，
            应同步记录到 assumptions
            或 open_questions。

        18. testing_strategy 必须对应本次需求的
            Acceptance Criteria 和主要风险。

        19. testing_strategy 当前描述：

                “需要证明什么”

            而不是提前决定：

                “具体修改哪个测试文件”。

        20. risks 只记录真实可能影响实施、
            稳定性或正确性的风险。

        21. 不要输出具体代码。

        22. 当前阶段只做技术设计，
            不要直接生成 Implementation Task。

        23. 当前 TechnicalSpec 后续还会经过：

                Logical Planning
                → Repository Retrieval
                → Investigation
                → Change Surface Resolution

            因此不要把当前技术假设
            提前固化为文件级执行事实。

        24. 只返回 JSON，
            不返回 Markdown。

        25. 返回的内容必须是严格合法的 JSON。

            特别注意：

            - JSON 字符串内部的双引号必须正确转义；
            - 数组元素之间必须有逗号；
            - 对象字段之间必须有逗号；
            - 所有 [] 和 {{}} 必须正确闭合；
            - 不允许 JavaScript / Python 风格注释；
            - 不允许 trailing comma。

        输出必须符合以下 JSON Schema：

        {json.dumps(
            schema,
            ensure_ascii=False,
        )}
        """.strip()

        input_data = {
            "requirement": (requirement.model_dump(mode="json")),
            "project_context": (project_context.model_dump(mode="json")),
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

        raw_output = response.content

        last_error: ValidationError | None = None

        # ==================================
        # Structured Output Boundary
        #
        # 第一次：
        #   正常解析
        #
        # 如果失败：
        #   把真实 Validation Error
        #   返回给模型做有限修复。
        #
        # max_validation_retries=1 表示：
        #
        #   1 次正常生成
        #   +
        #   最多 1 次修复
        # ==================================

        for attempt in range(self.max_validation_retries + 1):
            try:
                return self._parse_spec(raw_output)

            except ValidationError as exc:
                last_error = exc

                if attempt >= self.max_validation_retries:
                    break

                raw_output = await self._repair_output(
                    raw_output=(raw_output),
                    error=exc,
                    schema=schema,
                )

        # ==================================
        # Repair 之后仍不合法
        #
        # 不降级、不猜测、不吞掉错误。
        # ==================================

        assert last_error is not None

        raise last_error

    @classmethod
    def _parse_spec(
        cls,
        raw_output: str,
    ) -> TechnicalSpec:
        """
        对 LLM 输出建立严格的
        Structured Output Boundary。

        这里只做非常轻量的外层清理：

        - 去掉首尾空白
        - 去掉 Markdown code fence
        - 截取最外层 JSON object

        不主动修复 JSON 内容。
        真正修复交给 bounded repair。
        """

        json_text = cls._extract_json_text(raw_output)

        return TechnicalSpec.model_validate_json(json_text)

    @staticmethod
    def _extract_json_text(
        raw_output: str,
    ) -> str:
        text = (raw_output or "").strip()

        if not text:
            # 为了统一进入
            # Pydantic ValidationError，
            # 给一个必然非法的 JSON。
            text = ""

        # ==================================
        # 容忍模型偶尔返回：
        #
        # ```json
        # {...}
        # ```
        #
        # 这里只去掉包装，
        # 不修改 JSON 本身。
        # ==================================

        if text.startswith("```"):
            lines = text.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        # ==================================
        # 有时模型会输出：
        #
        # Here is the result:
        # {...}
        #
        # Prompt 已禁止这种行为，
        # 这里仍做有限容错。
        # ==================================

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end >= start:
            text = text[start : end + 1]

        return text

    async def _repair_output(
        self,
        *,
        raw_output: str,
        error: ValidationError,
        schema: dict,
    ) -> str:
        """
        只修复 Structured Output。

        不是重新做技术方案，
        也不允许扩大需求范围。
        """

        repair_request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content=f"""
你正在修复一个 TechnicalSpec JSON。

上一份输出没有通过程序的严格校验。

你的任务只是修复 JSON 语法或 Schema 契约问题，
不要重新设计技术方案，
不要扩大需求范围，
不要添加上一份输出中不存在的新功能。

必须遵守：

1. 只返回一个合法 JSON object。
2. 不要返回 Markdown。
3. 不要返回 ```json code fence。
4. 不要解释修改过程。
5. 不要添加 JSON Schema 之外的字段。
6. 字段类型必须严格符合 Schema。
7. 数组和对象必须使用合法 JSON 语法。
8. JSON 字符串内部的双引号必须正确转义。
9. 尽可能保留上一份输出的原始语义。
10. 如果只是 JSON 格式错误，只修格式，不改变内容。

JSON Schema：

{json.dumps(
    schema,
    ensure_ascii=False,
)}
""".strip(),
                ),
                Message(
                    role="user",
                    content=json.dumps(
                        {
                            "validation_error": (self._format_validation_error(error)),
                            "previous_output": (raw_output),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
            ]
        )

        response = await self.llm.chat(repair_request)

        return response.content

    @staticmethod
    def _format_validation_error(
        error: ValidationError,
    ) -> str:
        """
        给 Repair LLM 一个紧凑但明确的
        Validation Error。

        不需要把完整 traceback
        塞回上下文。
        """

        details: list[str] = []

        for item in error.errors(include_url=False):
            location = ".".join(
                str(part)
                for part in item.get(
                    "loc",
                    (),
                )
            )

            message = item.get(
                "msg",
                "validation error",
            )

            error_type = item.get(
                "type",
                "unknown",
            )

            if location:
                details.append(f"{location}: " f"{message} " f"({error_type})")
            else:
                details.append(f"{message} " f"({error_type})")

        return "\n".join(details)
