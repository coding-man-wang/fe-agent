from typing import Literal

from pydantic import BaseModel, Field


class RequirementSpec(BaseModel):
    title: str = Field(description="需求标题，简短描述本次要开发的功能")

    summary: str = Field(description="对用户原始需求的结构化总结")

    goals: list[str] = Field(
        default_factory=list, description="本次需求需要实现的核心目标"
    )

    non_goals: list[str] = Field(
        default_factory=list, description="明确本次需求不处理的内容"
    )

    acceptance_criteria: list[str] = Field(
        default_factory=list, description="判断需求是否完成的验收标准"
    )

    constraints: list[str] = Field(
        default_factory=list, description="开发过程中必须遵守的限制和约束"
    )

    assumptions: list[str] = Field(
        default_factory=list, description="当前分析过程中暂时采用的假设"
    )

    open_questions: list[str] = Field(
        default_factory=list, description="由于信息不足，目前仍需要确认的问题"
    )


class RequirementIssue(BaseModel):
    # missing 信息缺失（未明确格式）
    # ambiguous 描述有歧义
    # conflict 需求冲突
    # unverifiable 无法验收
    # scope 范围问题
    category: Literal["missing", "ambiguous", "conflict", "unverifiable", "scope"]

    severity: Literal[
        "warning",
        "blocking",
    ]

    message: str

    related_field: str | None = None


class RequirementValidationResult(BaseModel):
    # 当前需求是否已经准备好进入下一环节
    is_ready: bool

    # 检查发现的问题
    issues: list[RequirementIssue] = Field(default_factory=list)

    # 需要向用户确认的问题
    clarification_questions: list[str] = Field(
        default_factory=list,
    )


class StringListPatch(BaseModel):
    add: list[str] = Field(default_factory=list, description="本次需要新增的内容")

    remove: list[str] = Field(
        default_factory=list, description="本次需要删除的原有内容"
    )


class RequirementPatch(BaseModel):
    title: str | None = Field(
        default=None, description="如果需求标题需要修改，填写新的标题，否则为 null"
    )

    summary: str = Field(
        default=None, description="如果需求总结需要修改，填写新的总结，否则为 null"
    )

    goals: StringListPatch = Field(
        default_factory=StringListPatch,
    )

    non_goals: StringListPatch = Field(
        default_factory=StringListPatch,
    )

    acceptance_criteria: StringListPatch = Field(
        default_factory=StringListPatch,
    )

    constraints: StringListPatch = Field(
        default_factory=StringListPatch,
    )

    assumptions: StringListPatch = Field(
        default_factory=StringListPatch,
    )

    open_questions: StringListPatch = Field(
        default_factory=StringListPatch,
    )


class ProjectContext(BaseModel):
    project_name: str = Field(description="项目名称")

    framework: str = Field(description="当前项目使用的前端框架，例如 Vue3、React")

    language: str = Field(description="主要开发语言，例如 TypeScript")

    build_tool: str | None = Field(
        default=None, description="项目构建工具，例如 Vite、Webpack"
    )

    package_manager: str | None = Field(
        default=None, description="包管理工具，例如 pnpm、npm、yarn"
    )

    state_management: str | None = Field(
        default=None, description="项目使用的状态管理方案，例如 Pinia"
    )

    test_stack: list[str] = Field(
        default_factory=list, description="项目当前测试技术栈"
    )

    available_components: list[str] = Field(
        default_factory=list,
        description="项目或公司内部已有、可以复用的组件",
    )

    engineering_rules: list[str] = Field(
        default_factory=list, description="当前项目必须遵守的工程规范"
    )

    directory_summary: list[str] = Field(
        default_factory=list, description="与当前开发相关的目录结构摘要"
    )

    relevant_files: list[str] = Field(
        default_factory=list, description="已经确定与当前需求相关的代码文件"
    )

    def repository_hint_candidates(
        self,
    ) -> list[str]:
        """
        返回前置 Planning 阶段允许引用的
        Repository Hint。

        这些 Hint 只能来自已经明确提供的
        ProjectContext Evidence。

        directory_summary 例如：

            src/api - API 访问层

        只暴露：

            src/api

        而不是让 Planner 自己推导：
            src/api/orders.ts
        """

        result: list[str] = []

        seen: set[str] = set()

        def append(
            value: str,
        ) -> None:

            normalized = (
                value.replace(
                    "\\",
                    "/",
                )
                .strip()
                .strip("/")
            )

            if not normalized:
                return

            key = normalized.lower()

            if key in seen:
                return

            seen.add(key)

            result.append(normalized)

        for entry in self.directory_summary:

            value = entry.strip()

            if not value:
                continue

            repository_hint = value

            for separator in (
                " - ",
                " – ",
                " — ",
            ):
                if separator in value:
                    repository_hint = value.split(
                        separator,
                        1,
                    )[0].strip()
                    break

            append(repository_hint)

        for path in self.relevant_files:
            append(path)

        return result


class TechnicalDecision(BaseModel):
    topic: str = Field(description="本次技术决策关注的问题")

    decision: str = Field(description="最终采用的技术方案")

    rationale: str = Field(description="为什么采用这个方案")


class TechnicalSpec(BaseModel):
    summary: str = Field(description="本次技术方案整体摘要")

    solution_approach: list[str] = Field(
        default_factory=list,
        description="整体实现方案和主要技术步骤",
    )

    affected_modules: list[str] = Field(
        default_factory=list,
        description="预计受到影响的项目模块",
    )

    reuse_plan: list[str] = Field(
        default_factory=list,
        description="可以直接复用的现有组件、工具或基础设施",
    )

    data_flow: list[str] = Field(
        default_factory=list,
        description="核心数据如何在系统中流转",
    )

    technical_decisions: list[TechnicalDecision] = Field(
        default_factory=list,
        description="本次方案中的关键技术决策及原因",
    )

    testing_strategy: list[str] = Field(
        default_factory=list,
        description="如何验证本次技术实现",
    )

    risks: list[str] = Field(
        default_factory=list,
        description="实施过程中可能存在的主要风险",
    )

    assumptions: list[str] = Field(
        default_factory=list,
        description="技术方案目前依赖但尚未完全确认的假设",
    )

    open_questions: list[str] = Field(
        default_factory=list,
        description="技术设计阶段仍然需要确认的问题",
    )


class ImplementationStep(BaseModel):
    order: int = Field(
        ge=1,
        description="实施步骤顺序，从 1 开始",
    )

    title: str = Field(description="当前实施步骤的简短标题")

    goal: str = Field(description="当前步骤希望达成的目标")

    actions: list[str] = Field(
        default_factory=list,
        description="当前步骤需要完成的主要动作",
    )

    affected_modules: list[str] = Field(
        default_factory=list,
        description="当前步骤预计影响的项目模块",
    )

    verification: list[str] = Field(
        default_factory=list,
        description="如何确认当前步骤已经正确完成",
    )


class ImplementationPlan(BaseModel):
    summary: str = Field(description="整个实施计划的总体说明")

    preconditions: list[str] = Field(
        default_factory=list,
        description="正式实施前必须满足的前置条件",
    )

    steps: list[ImplementationStep] = Field(
        default_factory=list,
        description="按照执行顺序排列的实施步骤",
    )

    completion_criteria: list[str] = Field(
        default_factory=list,
        description="整个实施计划完成的判断标准",
    )

    risks: list[str] = Field(
        default_factory=list,
        description="实施过程中需要重点关注的风险",
    )

    open_questions: list[str] = Field(
        default_factory=list,
        description="当前实施计划仍未解决的问题",
    )


TaskType = Literal[
    "investigation",
    "code_change",
    "test",
    "verification",
]

TaskPlanningStage = Literal[
    "legacy",
    "logical",
]


class ExecutionObligation(BaseModel):
    """
    Runtime Execution Plan 中必须兑现的一项逻辑任务义务。

    它不是 Planner 直接生成的内容。

    Logical TaskPlan:
        execution_obligations = []

    ExecutionPlanCompressor:
        根据被聚合的 Logical Tasks
        生成 execution_obligations。

    Harness 用它保证：

        Execution Granularity 可以变粗，

    但：

        Logical Responsibility / Acceptance
        不能因此消失。
    """

    obligation_id: str = Field(
        min_length=1,
        description="稳定的执行义务 ID，当前使用来源 Task ID",
    )

    source_task_id: str = Field(
        min_length=1,
        description="该执行义务来源的 Logical Task",
    )

    objective: str = Field(
        min_length=1,
        description="该 Logical Task 原始目标",
    )

    responsibilities: list[str] = Field(
        default_factory=list,
        description="必须完成的工程责任",
    )

    expected_outputs: list[str] = Field(
        default_factory=list,
        description="完成该义务应该产生的结果",
    )

    acceptance_criteria_refs: list[str] = Field(
        default_factory=list,
        description="该义务对应的 Requirement Acceptance Criteria",
    )


class TaskDefinition(BaseModel):
    task_id: str = Field(description="任务唯一编号，例如 TASK-001")

    task_type: TaskType = Field(description="任务类型")

    title: str = Field(description="任务标题")

    objective: str = Field(description="这个任务完成后应该达成什么目标")

    source_step_order: int = Field(
        ge=1, description="这个任务来源于 ImplementationPlan 的第几步"
    )

    depends_on: list[str] = Field(
        default_factory=list,
        description="当前任务依赖的其他 task_id",
    )

    inputs: list[str] = Field(
        default_factory=list,
        description=("执行当前任务需要参考的" "结构化输入或上游结果"),
    )

    # ==================================
    # v1.3 Logical Planning
    #
    # 前置 TaskPlan 负责：
    #
    # “解决什么责任问题”
    #
    # 而不是：
    #
    # “修改哪个具体文件”
    # ==================================

    responsibilities: list[str] = Field(
        default_factory=list,
        description=(
            "当前 Task 负责解决的"
            "逻辑/行为/技术责任；"
            "不得把未经 Repository "
            "Evidence 确认的具体文件"
            "当作责任描述"
        ),
    )

    investigation_questions: list[str] = Field(
        default_factory=list,
        description=(
            "需要通过 Repository "
            "Investigation 回答的问题；"
            "这些问题用于指导 Retrieval "
            "和后续 Repository Evidence 收集"
        ),
    )

    repository_hints: list[str] = Field(
        default_factory=list,
        description=(
            "前置 Planning 阶段允许使用的"
            "低承诺 Repository Hint；"
            "只能来自 ProjectContext 已明确"
            "提供的目录或文件信息，"
            "不能由 Planner 自行猜路径"
        ),
    )

    # ==================================
    # Legacy Compatibility
    #
    # 不删除。
    #
    # 原因：
    #
    # - 历史 Benchmark Run
    # - PreparedExecutionInput
    # - 已冻结 TaskPlan
    #
    # 仍然可能包含 target_modules。
    #
    # v1.3 新 Logical Plan
    # 必须输出 []。
    # ==================================

    target_modules: list[str] = Field(
        default_factory=list,
        description=(
            "v1.2 兼容字段。"
            "v1.3 新生成的 logical plan "
            "必须保持为空；"
            "只用于读取历史冻结 TaskPlan"
        ),
    )

    expected_outputs: list[str] = Field(
        default_factory=list,
        description="任务执行后应该产生的结果",
    )

    verification: list[str] = Field(
        default_factory=list,
        description="判断当前任务是否完成的方法",
    )

    acceptance_criteria_refs: list[str] = Field(
        default_factory=list,
        description="当前任务覆盖的 Requirement Acceptance Criteria",
    )

    execution_obligations: list[ExecutionObligation] = Field(
        default_factory=list,
        description=(
            "Runtime Execution Plan "
            "由 Harness/Compressor 注入的执行义务；"
            "Logical Planner 必须保持为空"
        ),
    )

    def repository_search_hints(
        self,
    ) -> list[str]:
        """
        Repository Retrieval 使用的
        统一 Hint 入口。

        新 Plan：
            repository_hints

        历史 Plan：
            target_modules

        新字段优先，
        旧字段只负责兼容。
        """

        result: list[str] = []

        seen: set[str] = set()

        for value in [
            *self.repository_hints,
            *self.target_modules,
        ]:

            normalized = (
                value.replace(
                    "\\",
                    "/",
                )
                .strip()
                .strip("/")
            )

            if not normalized:
                continue

            key = normalized.lower()

            if key in seen:
                continue

            seen.add(key)

            result.append(normalized)

        return result


class TaskPlan(BaseModel):

    planning_stage: TaskPlanningStage = Field(
        default="legacy",
        description=(
            "TaskPlan 所处 Planning 阶段。"
            "legacy 表示历史 Plan；"
            "logical 表示 v1.3 "
            "Investigation 前的低承诺逻辑计划"
        ),
    )

    summary: str = Field(description=("整个 Task Plan 的总体说明"))

    tasks: list[TaskDefinition] = Field(
        default_factory=list,
        description=("所有逻辑任务"),
    )

    completion_criteria: list[str] = Field(
        default_factory=list,
        description=("整个任务计划完成标准"),
    )
