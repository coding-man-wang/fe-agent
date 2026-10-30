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


class CodingEvidenceError(CodingOutputError):
    """
    Finding 引用的 evidence
    无法被对应 Code Context 证明。

    这是一个可进行一次
    Bounded Evidence Correction
    的窄错误类型。

    注意：
    - 文件不在 Context 中，不属于此错误；
    - Scope 越界，不属于此错误；
    - task_id 错误，不属于此错误；
    - proposed change 非法，不属于此错误。
    """


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
      "relative_path": "需要修改或创建的文件路径",
      "action": "modify 或 create",
      "objective": "修改目标",
      "proposed_approach": "建议修改方式"
    }
  ],
  "verification_plan": [
    "建议执行的验证步骤"
  ],
  "missing_information": [],
  "evidence_gaps": [
    {
      "description": "缺少的具体信息",
      "severity": "blocking 或 advisory",
      "resolution_stage": "analysis 或 verification",
      "related_paths": [
        "相关文件路径"
      ],
      "reason": "为什么缺少该信息，以及应该在哪个阶段解决"
    }
  ],
  "execution_status": "not_executed"
  "obligation_coverage": [
      {
        "obligation_id": "TASK-005",
        "status": "addressed_by_proposed_change",
        "related_paths": [
          "src/views/OrdersView.vue"
        ],
        "reason": "本次 proposal 将实现该 Logical Task 对应责任"
      }
    ]
}

要求：
1. findings 只能引用提供的 Code Context
   中真实存在的文件。

2. action=modify：
   relative_path 必须来自 Code Context，
   并且必须属于当前 Change Surface
   的 writable targets。

3. action=create：
   relative_path 不需要出现在 Code Context，
   但必须严格来自
   change_surface.create_files。

4. 不得为 create target 伪造 findings.evidence，
   因为该文件当前还不存在。

5. create target 当前没有源码内容，
   本身不属于 blocking evidence gap。

6. 如果 Repository test_stack / scripts
   已经能确认测试框架和运行方式，
   “当前没有同功能测试文件”
   不能单独作为 blocking evidence gap。

7. 已有测试文件如果只是作为模式参考，
   不得因为读取了它就修改它。
8. findings.evidence 必须直接复制自相应文件的 code_context.content。

   - 优先引用短小、明确、稳定的源码片段；
   - 最好引用单行源码；
   - 不要使用 Markdown ```；
   - 不要使用 ... 省略代码；
   - 不要翻译代码；
   - 不要修改变量名、函数名、字符串或运算符；
   - 不要为了美观主动重写源码。

   如果必须引用多行代码，可以改变换行和缩进，
   但实际代码内容必须来自提供的 code_context。
9. 如果代码不足以支持结论，不要编造证据；
   应放入 missing_information。
10. 只提出修改方案，不声称已经修改或测试。
11. 不要输出工具调用或可执行动作指令。
12. 新输出优先使用 evidence_gaps，
   missing_information 保持为空数组。

13. severity = blocking：
   缺少该信息会导致你无法可靠判断：
   - 当前 implementation target 是否需要修改；
   - 应该如何安全修改；
   - 修改是否违反明确接口、业务或架构约束。

14. severity = advisory：
   信息虽然有帮助，但不影响当前是否需要修改、
   或如何修改的判断。

15. resolution_stage = analysis：
   该证据必须通过进一步分析代码、接口、配置、
   业务规则等方式补齐。

   例如：
   - implementation target 的关键代码未提供；
   - API contract 不明确；
   - 无法确认关键业务约束；
   - 无法判断应该修改哪个模块。

16. resolution_stage = verification：
    当前静态代码已经足够判断是否需要修改以及如何修改，
    但还需要后续执行来证明行为正确。

    例如：
    - 尚未执行 unit test；
    - 尚未执行 build；
    - 尚未获取 runtime log；
    - 尚未获取网络请求记录；
    - 尚未实际验证是否存在重复请求；
    - 尚未实际验证 UI interaction。

17. 不要因为尚未运行测试、构建、浏览器或运行时验证，
    就阻塞当前 Coding Decision。

    如果这些信息只用于证明修改后的行为是否正确：
    severity 应为 advisory，
    resolution_stage 应为 verification。

18. 只有当缺少运行时信息导致你连
    “是否需要修改”或“应该如何修改”
    都无法判断时，
    才可以使用：
    resolution_stage = analysis。

19. resolution_stage = verification 的 Evidence Gap
    不应被用来扩大 proposed_changes。
    Verification 负责证明行为，
    Coding Analysis 负责决定修改。
20. 不要基于未提供证据的“假设性项目约定”
    创建 blocking evidence gap。

    例如：
    “如果项目约定必须使用某个常量……”
    只有当 Code Context、Requirement、
    Engineering Rules 或 Upstream Artifacts
    中存在该约定证据时，才能作为 blocker。

21. 如果你已经明确判断某项缺失信息
    “不影响当前任务判断”、
    “不影响是否需要修改”或
    “当前代码已经足够判断”，
    severity 不允许为 blocking。

22. 当前 Task 可能运行在前序任务已经修改后的
    最新 Repository Revision 上。

    如果当前最新代码已经满足本 Task，
    应返回：
    proposed_changes = []
    并且不要为了执行原计划而重复修改代码。
23. 如果 task.execution_obligations 非空，
    obligation_coverage 必须逐项覆盖所有 obligation。

24. 每个 execution_obligation 必须且只能返回一次。

25. status=satisfied_by_current_code：
    related_paths 必须引用当前 Code Context
    中真实存在的文件，并且 findings 必须包含
    至少一个对应路径的源码 Evidence。

26. status=addressed_by_proposed_change：
    related_paths 必须属于 proposed_changes。

27. status=unresolved：
    表示当前 Execution Unit 无法安全证明
    该 Logical Task 已经得到处理。
    Harness 会阻止进入 Patch。

28. 禁止因为多个 execution_obligations
    修改同一个文件，就只覆盖其中一个 obligation。

29. execution_obligations 描述的是
    “必须完成什么”，
    Change Surface 仍决定
    “当前允许修改什么”。

30. 不得通过扩大 proposed_changes
    来满足 execution_obligations；
    所有修改仍必须严格位于
    Change Surface writable boundary 内。
31. 如果当前 Task 是 test，
    并且 verification_strategy 不为空，
    proposed_changes / proposed_approach
    必须遵守 verification_strategy。

32. 对 routing_action=required_here 的 Proof：
    当前测试方案应该明确覆盖。

33. 对 routing_action=deferred 的 Proof：
    不得为了当前 Test Task 看起来完整，
    在不可观察的 Test Layer 中伪造 Proof Strategy。

34. verification_strategy.forbidden_strategies
    中声明的能力不能进入 proposed_approach。

35. 如果当前 E2E Runtime Path 是 local_mock，
    不得因为源码中还存在 Remote 分支，
    就假设 Browser 当前一定会发送 Remote HTTP Request。

36. “当前层不可观察”
    不是 blocking coding evidence gap。
    这是 Proof Routing 问题，
    不应该阻塞当前层对其他可观察行为的测试实现。
""".strip()

    RUNTIME_RECONCILIATION_REVIEW_INSTRUCTION = """
    你正在进行 Runtime Reconciliation Review。

    当前 Repository Context 来自前序任务已经提交后的最新 Revision。
    因此当前 Task 原计划要求的修改，可能已经被上游 Task 完成。

    请重新判断：

    1. 当前最新代码是否仍然需要任何修改。
    2. 如果当前代码已经满足本 Task：
       - proposed_changes 必须为空；
       - 不要为了避免 noop 而制造新的 blocking evidence gap。

    3. blocking evidence gap 只有在缺少该信息会导致你无法判断：
       - 当前代码是否还需要修改；
       - 或剩余修改应该如何安全完成；
       时才允许存在。

    4. 不要基于没有证据的假设性项目约定制造 blocker。

       例如：
       - “如果项目约定必须使用某个常量……”
       - “如果项目可能存在另外一种规范……”
       - “可能还有未提供的 helper……”

       如果当前 Code Context、Requirement、Engineering Rules、
       Change Surface 和 Upstream Artifacts 都没有提供这种约束证据，
       不得把这种假设作为 blocking evidence gap。

    5. 如果你在 description 或 reason 中已经判断：
       - “不影响当前任务判断”
       - “当前实现已经足够”
       - “不影响是否需要修改”
       那么该 Evidence Gap 不允许标记为 blocking。

    6. 如果当前静态代码已经足以判断无需继续修改：
       返回：
       - proposed_changes = []
       - evidence_gaps = []
         或仅保留真正 advisory 的 gap。

    7. 如果经过重新检查后仍然存在真实 blocker，
       请继续保留 blocking evidence gap。
       不要为了让流程通过而删除真实风险。

    仍然必须严格遵守原始 Coding Analysis JSON 契约。
    """.strip()

    @staticmethod
    def _normalize_evidence_layout(
        text: str,
    ) -> str:
        """
        只对 Evidence 的“展示格式”做归一化。

        允许：

            原源码：
            async function foo() { doA(); doB() }

            模型引用：
            async function foo() {
              doA();
              doB()
            }

        不允许模型随意修改：

            - 函数名
            - 变量名
            - 字符串内容
            - 运算符
            - 实际代码字符

        这里刻意不使用：

            re.sub(r"\\s+", "", text)

        因为那样会把字符串内部的空格也删除，
        可能错误接受并不存在的代码。
        """

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        lines = [line.strip() for line in normalized.split("\n")]

        return " ".join(line for line in lines if line)

    @classmethod
    def _evidence_exists(
        cls,
        *,
        evidence: str,
        content: str,
    ) -> bool:
        """
        两级 Evidence 校验。

        第一级：
            严格原文匹配。

        第二级：
            仅忽略换行 / 缩进造成的版式差异。

        这样既尽量保持证据真实性，
        又避免模型只是重新排版代码就导致整个 Task 失败。
        """

        # 最可靠情况：
        # 模型直接复制了源码原文。
        if evidence in content:
            return True

        normalized_evidence = cls._normalize_evidence_layout(evidence)

        if not normalized_evidence:
            return False

        normalized_content = cls._normalize_evidence_layout(content)

        return normalized_evidence in normalized_content

    @staticmethod
    def _format_validation_errors(
        exc: ValidationError,
    ) -> str:
        """
        把 Pydantic 的结构校验错误转换成
        更适合人和 LLM 阅读的简洁信息。

        例如：

            evidence_gaps.0.severity:
            Input should be 'blocking' or 'advisory'
        """

        lines: list[str] = []

        for error in exc.errors(include_url=False):
            location = ".".join(str(part) for part in error.get("loc", ()))

            message = str(
                error.get(
                    "msg",
                    "validation error",
                )
            )

            error_type = str(
                error.get(
                    "type",
                    "unknown",
                )
            )

            if location:
                lines.append(f"{location}: " f"{message} " f"[{error_type}]")
            else:
                lines.append(f"{message} " f"[{error_type}]")

        return "\n".join(lines)

    async def _repair_invalid_output(
        self,
        *,
        messages: RenderedContext,
        validation_error: ValidationError,
    ) -> str:
        """
        第一次模型输出无法通过
        ReadOnlyCodingResult 结构校验时，
        给模型一次重新输出机会。

        注意：

        这里只允许重新生成结构化分析结果。

        此时：
        - 没有修改文件
        - 没有运行命令
        - 没有副作用

        第二次输出仍然要重新经过：
        1. Pydantic Schema 校验
        2. Evidence 源码真实性校验
        3. 文件范围校验
        """

        error_details = self._format_validation_errors(validation_error)

        repair_instruction = f"""
    你上一次返回的 JSON 无法通过结构校验。

    请重新阅读原始 Task Context，
    重新返回一个完整的合法 JSON 对象。

    不要输出解释。
    不要输出 Markdown 代码块。
    不要输出 JSON 之外的任何文本。

    必须继续遵守原来的所有 Coding Analysis 规则。

    尤其注意：

    1. severity 只能是：
       - blocking
       - advisory

    2. resolution_stage 只能是：
       - analysis
       - verification

    3. execution_status 必须是：
       - not_executed

    4. 不要增加契约中不存在的字段。

    5. findings.evidence 必须来自提供的 code_context。

    6. action=modify 只能引用提供的代码文件。

    7. action=create 只能引用
       change_surface.create_files。
    
    8. create target 当前没有 code_context
       是正常情况，不要因此改成 blocking gap。

    上一次结构校验错误：

    {error_details}
    """.strip()

        return await self.model.complete(
            system_message=(messages.system_message + "\n\n" + repair_instruction),
            user_message=(messages.user_message),
        )

    @staticmethod
    def _validate_evidence_correction_boundary(
        *,
        original: ReadOnlyCodingResult,
        corrected: ReadOnlyCodingResult,
    ) -> None:
        """
        Evidence Correction 不是第二次
        Coding Analysis。

        唯一允许变化的是：

            findings[*].evidence

        其他业务判断必须保持不变。

        这样可以避免模型利用
        “纠正证据”偷偷进行：

        - Replan
        - Scope Expansion
        - 删除 Change
        - 修改 Evidence Gap
        - 修改 Verification Plan
        """

        if original.task_id != corrected.task_id:
            raise CodingOutputError("Evidence Correction " "不得修改 task_id")

        if original.summary != corrected.summary:
            raise CodingOutputError("Evidence Correction " "不得修改 summary")

        if len(original.findings) != len(corrected.findings):
            raise CodingOutputError("Evidence Correction " "不得增加或删除 Finding")

        for index, (
            original_finding,
            corrected_finding,
        ) in enumerate(
            zip(
                original.findings,
                corrected.findings,
                strict=True,
            )
        ):
            if original_finding.relative_path != corrected_finding.relative_path:
                raise CodingOutputError(
                    "Evidence Correction "
                    "不得修改 Finding 文件路径; "
                    f"index={index}"
                )

            if original_finding.explanation != corrected_finding.explanation:
                raise CodingOutputError(
                    "Evidence Correction " "不得修改 Finding 结论; " f"index={index}"
                )

        if original.proposed_changes != corrected.proposed_changes:
            raise CodingOutputError("Evidence Correction " "不得修改 proposed_changes")

        if original.verification_plan != corrected.verification_plan:
            raise CodingOutputError("Evidence Correction " "不得修改 verification_plan")

        if original.missing_information != corrected.missing_information:
            raise CodingOutputError(
                "Evidence Correction " "不得修改 missing_information"
            )

        if original.evidence_gaps != corrected.evidence_gaps:
            raise CodingOutputError("Evidence Correction " "不得修改 evidence_gaps")

        if original.execution_status != corrected.execution_status:
            raise CodingOutputError("Evidence Correction " "不得修改 execution_status")

    async def _repair_invalid_evidence(
        self,
        *,
        messages: RenderedContext,
        previous_result: ReadOnlyCodingResult,
        evidence_error: CodingEvidenceError,
    ) -> str:
        """
        对已经通过 Schema Validation、
        但 Evidence Authenticity Validation
        失败的结果进行一次有限纠正。

        这里只修 findings.evidence。

        不允许：
        - 改业务判断；
        - 改 proposed_changes；
        - 改 Scope；
        - 增删 Finding；
        - 改 Finding path / explanation。
        """

        previous_json = previous_result.model_dump_json(indent=2)

        correction_instruction = f"""
    你正在执行一次：

    Bounded Evidence Correction

    上一轮 Coding Analysis 的 JSON
    已经通过 Schema 校验，

    但是其中至少一个：

        findings[*].evidence

    无法在对应的 Code Context 中找到。

    这意味着：

        分析结果的结构合法，
        但源码引用不真实或不精确。

    你的任务不是重新分析代码。

    你只允许纠正：

        findings[*].evidence

    硬性规则：

    1. 必须返回完整 JSON 对象。

    2. findings 数量不能变化。

    3. findings 顺序不能变化。

    4. relative_path 不能变化。

    5. explanation 不能变化。

    6. summary 不能变化。

    7. proposed_changes 不能变化。

    8. verification_plan 不能变化。

    9. missing_information 不能变化。

    10. evidence_gaps 不能变化。

    11. execution_status 不能变化。

    12. task_id 不能变化。

    13. 唯一允许修改的字段是：

            findings[*].evidence

    14. evidence 必须直接复制自
        对应 relative_path 的
        code_context.content。

    15. 优先引用：

        - 单行代码；
        - 短小片段；
        - 稳定源码；
        - 能直接支持原 explanation
          的最小证据。

    16. 不得：
        - 猜测代码；
        - 改变量名；
        - 改字符串；
        - 使用 ...；
        - 翻译源码；
        - 根据记忆补全源码。

    17. 如果无法找到能支持原 Finding
        的真实 Evidence：

        不允许修改 Finding 结论来绕过校验。

        返回结果仍必须保持原 Decision。
        Harness 会再次校验并 Fail Closed。

    上一轮 Evidence Validation Error：

    {str(evidence_error)}

    上一轮结构化结果：

    {previous_json}
    """.strip()

        return await self.model.complete(
            system_message=(messages.system_message + "\n\n" + correction_instruction),
            user_message=(messages.user_message),
        )

    @staticmethod
    def _parse_result(
        raw_output: str,
    ) -> ReadOnlyCodingResult:
        return ReadOnlyCodingResult.model_validate_json(raw_output)

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

            if not self._evidence_exists(
                evidence=finding.evidence,
                content=content,
            ):
                evidence_preview = finding.evidence.replace("\n", "\\n")

                if len(evidence_preview) > 180:
                    evidence_preview = evidence_preview[:180] + "..."

                raise CodingEvidenceError(
                    "Finding 的 evidence "
                    "不存在于对应代码上下文: "
                    f"{finding.relative_path}; "
                    f"evidence={evidence_preview!r}"
                )

        surface_data = bundle.change_surface or {}

        create_paths = set(
            surface_data.get(
                "create_files",
                [],
            )
        )

        writable_paths = {
            *surface_data.get(
                "implementation_targets",
                [],
            ),
            *surface_data.get(
                "tests",
                [],
            ),
        }

        for change in result.proposed_changes:

            path = change.relative_path

            if path not in writable_paths:
                raise CodingOutputError(
                    "修改提案超出 " "Change Surface " "写边界: " f"{path}"
                )

            if change.action == "modify":
                if path not in code_map:
                    raise CodingOutputError(
                        "modify 提案引用了" "未提供的已有文件: " f"{path}"
                    )

                if path in create_paths:
                    raise CodingOutputError(
                        "create target " "不能声明为 modify: " f"{path}"
                    )

                continue

            if change.action == "create":
                if path not in create_paths:
                    raise CodingOutputError(
                        "create 提案不属于 "
                        "change_surface."
                        "create_files: "
                        f"{path}"
                    )

                if path in code_map:
                    raise CodingOutputError(
                        "create target " "已经存在于 " "Code Context: " f"{path}"
                    )

                continue

            raise CodingOutputError("未知 proposed change " f"action: {change.action}")
        # ==================================
        # Execution Obligation Coverage
        # ==================================

        obligations = list(
            getattr(
                bundle.task,
                "execution_obligations",
                [],
            )
        )

        if not obligations:

            if result.obligation_coverage:
                raise CodingOutputError(
                    "当前 Task 不存在 "
                    "execution_obligations，"
                    "但模型返回了 "
                    "obligation_coverage"
                )

            return

        expected_ids = [item.obligation_id for item in obligations]

        actual_ids = [item.obligation_id for item in result.obligation_coverage]

        if len(actual_ids) != len(set(actual_ids)):
            raise CodingOutputError("obligation_coverage " "存在重复 obligation_id")

        missing_ids = sorted(set(expected_ids) - set(actual_ids))

        unexpected_ids = sorted(set(actual_ids) - set(expected_ids))

        if missing_ids:
            raise CodingOutputError(
                "Execution Obligation "
                "Coverage 不完整，缺少: " + ", ".join(missing_ids)
            )

        if unexpected_ids:
            raise CodingOutputError(
                "obligation_coverage "
                "包含未知 obligation: " + ", ".join(unexpected_ids)
            )

        finding_paths = {finding.relative_path for finding in result.findings}

        proposed_paths = {change.relative_path for change in result.proposed_changes}

        for coverage in result.obligation_coverage:

            if coverage.status == "satisfied_by_current_code":

                if not coverage.related_paths:
                    raise CodingOutputError(
                        f"{coverage.obligation_id} "
                        "声明当前代码已满足，"
                        "但没有提供 related_paths"
                    )

                invalid_paths = set(coverage.related_paths) - finding_paths

                if invalid_paths:
                    raise CodingOutputError(
                        f"{coverage.obligation_id} "
                        "声明当前代码已满足，"
                        "但以下路径没有 "
                        "Finding Evidence: " + ", ".join(sorted(invalid_paths))
                    )

                continue

            if coverage.status == ("addressed_by_" "proposed_change"):

                if not coverage.related_paths:
                    raise CodingOutputError(
                        f"{coverage.obligation_id} "
                        "声明由 proposed change "
                        "完成，但 related_paths "
                        "为空"
                    )

                invalid_paths = set(coverage.related_paths) - proposed_paths

                if invalid_paths:
                    raise CodingOutputError(
                        f"{coverage.obligation_id} "
                        "引用了不属于 "
                        "proposed_changes "
                        "的路径: " + ", ".join(sorted(invalid_paths))
                    )

                continue

            if coverage.status == "unresolved":
                continue

            raise CodingOutputError(
                "未知 Execution Obligation " "Coverage Status: " f"{coverage.status}"
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
            system_message=(messages.system_message),
            user_message=(messages.user_message),
        )

        try:
            result = self._parse_result(raw_output)

        except ValidationError as first_error:

            # ==================================
            # 第一次结构不合法：
            #
            # 不直接让整个 Coding Task 失败。
            #
            # 给 LLM 一次针对结构错误的
            # 修正机会。
            # ==================================

            repaired_output = await self._repair_invalid_output(
                messages=messages,
                validation_error=(first_error),
            )

            try:
                result = self._parse_result(repaired_output)

            except ValidationError as second_error:

                error_details = self._format_validation_errors(second_error)

                # 只保留一小段方便调试，
                # 避免把整份模型输出塞进异常。
                output_preview = repaired_output.replace(
                    "\n",
                    "\\n",
                )

                if len(output_preview) > 500:
                    output_preview = output_preview[:500] + "..."

                raise CodingOutputError(
                    "模型连续两次输出都不符合 "
                    "ReadOnlyCodingResult 契约; "
                    f"errors={error_details}; "
                    f"output={output_preview!r}"
                ) from second_error

        # ==================================
        # Evidence Authenticity Boundary
        #
        # Schema 合法
        # !=
        # Repository Evidence 真实
        #
        # 第一次 Evidence 不合法时，
        # 只允许进行一次
        # Bounded Evidence Correction。
        # ==================================

        try:
            self._validate_result(
                bundle=bundle,
                result=result,
            )

        except CodingEvidenceError as first_error:

            corrected_output = await self._repair_invalid_evidence(
                messages=messages,
                previous_result=result,
                evidence_error=first_error,
            )

            # ------------------------------
            # Evidence Correction
            # 仍然必须重新经过 Schema。
            #
            # 这里不再进行第二层
            # Schema Repair，
            # 防止 Correction Chain
            # 无限扩张。
            # ------------------------------

            try:
                corrected_result = self._parse_result(corrected_output)

            except ValidationError as exc:

                error_details = self._format_validation_errors(exc)

                raise CodingOutputError(
                    "Evidence Correction " "返回了非法结构; " f"errors={error_details}"
                ) from exc

            # ------------------------------
            # Correction Integrity Guard
            #
            # Evidence Correction
            # 只能改变引用文本，
            # 不能改变 Coding Decision。
            # ------------------------------

            self._validate_evidence_correction_boundary(
                original=result,
                corrected=corrected_result,
            )

            # ------------------------------
            # 再次执行完整 Validation。
            #
            # 不仅重新检查 Evidence，
            # 也重新检查：
            #
            # - task_id
            # - Change Surface
            # - modify/create
            # - writable scope
            # ------------------------------

            try:
                self._validate_result(
                    bundle=bundle,
                    result=corrected_result,
                )

            except CodingEvidenceError as second_error:

                raise CodingOutputError(
                    "Bounded Evidence Correction "
                    "后 Evidence 仍然不真实: "
                    f"{second_error}"
                ) from second_error

            result = corrected_result

        return result

    def prepare_runtime_reconciliation_messages(
        self,
        messages: RenderedContext,
    ) -> RenderedContext:
        """
        当第一次 Coding Analysis：

            没有 proposed_changes
            +
            返回 blocking evidence gap

        时，进行一次有限的 Runtime
        Reconciliation Review。

        注意：

        这里只重新判断最新 Revision 是否
        仍然需要修改。

        不降低 Harness Safety Gate。
        """

        return RenderedContext(
            system_message=(
                messages.system_message
                + "\n\n"
                + self.RUNTIME_RECONCILIATION_REVIEW_INSTRUCTION
            ),
            user_message=(messages.user_message),
        )
