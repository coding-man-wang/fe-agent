from typing import Literal

from pydantic import BaseModel, Field


class RequirementSpec(BaseModel):
    title: str = Field(description="需求标题，简短描述本次要开发的功能")

    summary: str = Field(description="对用户原始需求的结构化总结")

    goals: list[str] = Field(
        default_factory=list,
        description="本次需求需要实现的核心目标"
    )

    non_goals: list[str] = Field(
        default_factory=list,
        description="明确本次需求不处理的内容"
    )

    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="判断需求是否完成的验收标准"
    )

    constraints: list[str] = Field(
        default_factory=list,
        description="开发过程中必须遵守的限制和约束"
    )

    assumptions: list[str] = Field(
        default_factory=list,
        description="当前分析过程中暂时采用的假设"
    )

    open_questions: list[str] = Field(
        default_factory=list,
        description="由于信息不足，目前仍需要确认的问题"
    )


class RequirementIssue(BaseModel):
    # missing 信息缺失（未明确格式）
    # ambiguous 描述有歧义
    # conflict 需求冲突
    # unverifiable 无法验收
    # scope 范围问题
    category: Literal[
        "missing",
        "ambiguous",
        "conflict",
        "unverifiable",
        "scope"
    ]

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
    add: list[str] = Field(
        default_factory=list,
        description="本次需要新增的内容"
    )

    remove: list[str] = Field(
        default_factory=list,
        description="本次需要删除的原有内容"
    )


class RequirementPatch(BaseModel):
    title: str | None = Field(
        default=None,
        description="如果需求标题需要修改，填写新的标题，否则为 null"
    )

    summary: str = Field(
        default=None,
        description="如果需求总结需要修改，填写新的总结，否则为 null"
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
    project_name: str = Field(
        description="项目名称"
    )

    framework: str = Field(
        description="当前项目使用的前端框架，例如 Vue3、React"
    )

    language: str = Field(
        description="主要开发语言，例如 TypeScript"
    )

    build_tool: str | None = Field(
        default=None,
        description="项目构建工具，例如 Vite、Webpack"
    )

    package_manager: str | None = Field(
        default=None,
        description="包管理工具，例如 pnpm、npm、yarn"
    )

    state_management: str | None = Field(
        default=None,
        description="项目使用的状态管理方案，例如 Pinia"
    )

    test_stack: list[str] = Field(
        default_factory=list,
        description="项目当前测试技术栈"
    )

    available_components: list[str] = Field(
        default_factory=list,
        description="项目或公司内部已有、可以复用的组件",
    )

    engineering_rules: list[str] = Field(
        default_factory=list,
        description="当前项目必须遵守的工程规范"
    )

    directory_summary: list[str] = Field(
        default_factory=list,
        description="与当前开发相关的目录结构摘要"
    )

    relevant_files: list[str] = Field(
        default_factory=list,
        description="已经确定与当前需求相关的代码文件"
    )


class TechnicalDecision(BaseModel):
    topic: str = Field(
        description="本次技术决策关注的问题"
    )

    decision: str = Field(
        description="最终采用的技术方案"
    )

    rationale: str = Field(
        description="为什么采用这个方案"
    )


class TechnicalSpec(BaseModel):
    summary: str = Field(
        description="本次技术方案整体摘要"
    )

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

    title: str = Field(
        description="当前实施步骤的简短标题"
    )

    goal: str = Field(
        description="当前步骤希望达成的目标"
    )

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
    summary: str = Field(
        description="整个实施计划的总体说明"
    )

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


class TaskDefinition(BaseModel):
    task_id: str = Field(
        description="任务唯一编号，例如 TASK-001"
    )

    task_type: TaskType = Field(
        description="任务类型"
    )

    title: str = Field(
        description="任务标题"
    )

    objective: str = Field(
        description="这个任务完成后应该达成什么目标"
    )

    source_step_order: int = Field(
        ge=1,
        description="这个任务来源于 ImplementationPlan 的第几步"
    )

    depends_on: list[str] = Field(
        default_factory=list,
        description="当前任务依赖的其他 task_id",
    )

    inputs: list[str] = Field(
        default_factory=list,
        description="执行当前任务需要参考的输入信息",
    )

    target_modules: list[str] = Field(
        default_factory=list,
        description="预计需要处理的项目模块",
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


class TaskPlan(BaseModel):
    summary: str = Field(
        description="整个 Task Plan 的总体说明"
    )

    tasks: list[TaskDefinition] = Field(
        default_factory=list,
        description="所有可执行任务"
    )

    completion_criteria: list[str] = Field(
        default_factory=list,
        description="整个任务计划完成标准",
    )
