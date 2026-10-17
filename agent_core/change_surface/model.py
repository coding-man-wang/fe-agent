from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

# ============================================================
# Investigation
# ============================================================


ObservedFileRole = Literal[
    "candidate_target",
    "dependency",
    "reference",
    "test",
    "unknown",
]


class ObservedRepositoryFile(BaseModel):
    """
    Investigation 阶段观察到的一个 Repository 文件。

    注意：

    这里描述的是 Observation，
    不是最终 Change Surface 决策。

    例如：

        src/stores/users.ts

    在 Investigation 中可能被认为：

        candidate_target

    但后面的 ChangeSurfaceResolver
    仍然可以根据更多 Evidence
    决定它最终是：

        implementation_target

    或：

        dependency

    ----------------------------------------------------------

    relative_path

        文件相对于 Repository Root 的路径。

    observed_role

        Investigation 阶段对文件用途的初步判断。

    evidence

        为什么认为这个文件具有这个角色。

        Evidence 应尽量描述事实，例如：

        - 定义 pageSize
        - setPageSize 会重置 page
        - fetchUsers 使用当前 query
        - UsersView 调用了该 store

        而不是只写：

        - "很相关"

    confidence

        当前 Observation 的置信度。

        0.0 ~ 1.0

        它不是模型概率的严格统计意义，
        只是 Harness 内部用于表达：

            strong evidence
            weak evidence

        的统一数值。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    relative_path: str = Field(
        min_length=1,
        description=("文件相对于 Repository Root 的路径"),
    )

    observed_role: ObservedFileRole = Field(
        description=("Investigation 阶段观察到的文件角色"),
    )

    evidence: list[str] = Field(
        default_factory=list,
        description=("支持当前 observed_role 判断的仓库事实"),
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=("当前 Investigation Observation 的置信度"),
    )

    @model_validator(
        mode="after",
    )
    def validate_evidence(
        self,
    ) -> "ObservedRepositoryFile":
        """
        unknown 可以没有 Evidence。

        但如果已经明确判断为：

            candidate_target
            dependency
            reference
            test

        就必须至少给出一条 Evidence。

        避免出现：

            role = candidate_target
            evidence = []

        这种不可解释结果。
        """

        if self.observed_role != "unknown" and not self.evidence:
            raise ValueError("非 unknown 的 observed_role " "必须至少包含一条 evidence")

        return self


class InvestigationArtifact(BaseModel):
    """
    Investigation Task 的结构化输出。

    解决的问题：

        上游 Investigation 到底发现了什么？

    它应该成为后续 Task 的真实输入，
    而不是把：

        "TASK-001 输出：xxx"

    重新当 Repository Search Query。

    ----------------------------------------------------------

    示例：

        TASK-001
            ↓
        Repository Retrieval
            ↓
        读取：
            UsersView.vue
            users.ts
            api/users.ts
            domain.ts
            ...
            ↓
        InvestigationArtifact

    后续 TASK-002 可以消费这个 Artifact。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    task_id: str = Field(
        min_length=1,
        description=("产生当前 Artifact 的 Investigation Task ID"),
    )

    summary: str = Field(
        min_length=1,
        description=("对本次 Repository Investigation " "结果的整体总结"),
    )

    observed_files: list[ObservedRepositoryFile] = Field(
        default_factory=list,
        description=("Investigation 中观察到的重要文件"),
    )

    findings: list[str] = Field(
        default_factory=list,
        description=("从 Repository 中确认的技术事实"),
    )

    unresolved_questions: list[str] = Field(
        default_factory=list,
        description=("当前 Repository Evidence " "仍无法解决的问题"),
    )

    @model_validator(
        mode="after",
    )
    def validate_unique_files(
        self,
    ) -> "InvestigationArtifact":
        """
        一个 Artifact 中同一个文件只保存一次。

        如果同一个文件同时有多类 Evidence，
        应该合并到同一个
        ObservedRepositoryFile.evidence 中。

        避免：

            users.ts → candidate_target
            users.ts → dependency

        这种结构歧义。
        """

        paths = [item.relative_path for item in self.observed_files]

        if len(paths) != len(set(paths)):
            raise ValueError("InvestigationArtifact " "observed_files 中存在重复文件")

        return self


# ============================================================
# Change Surface
# ============================================================


class ResolvedChangeSurface(BaseModel):
    """
    Change Surface Resolver 的结构化输出。

    InvestigationArtifact 表达：

        我看到了什么？

    ResolvedChangeSurface 表达：

        基于这些 Repository Facts，
        这个 Task 最终准备怎么使用这些文件？

    ----------------------------------------------------------

    implementation_targets

        真正候选修改文件。

    dependencies

        为了正确修改 implementation_target
        必须理解，但原则上不应该修改的文件。

    references

        类似实现 / 邻接实现 / 设计参考。

    tests

        与本次改动直接相关的测试文件。

    unresolved

        当前 Evidence 不足，
        还不能确定角色的文件或问题。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    task_id: str = Field(
        min_length=1,
        description=("当前 Change Surface 对应的 Task ID"),
    )

    implementation_targets: list[str] = Field(
        default_factory=list,
        description=("真正预计需要修改的实现文件"),
    )

    dependencies: list[str] = Field(
        default_factory=list,
        description=("需要理解但原则上不修改的依赖文件"),
    )

    references: list[str] = Field(
        default_factory=list,
        description=("类似实现或邻接实现参考文件"),
    )

    tests: list[str] = Field(
        default_factory=list,
        description=("与当前 Change Surface " "直接相关的测试文件"),
    )

    unresolved: list[str] = Field(
        default_factory=list,
        description=("Evidence 不足，尚未能完成分类的内容"),
    )

    @model_validator(
        mode="after",
    )
    def validate_file_roles(
        self,
    ) -> "ResolvedChangeSurface":
        """
        一个文件不应该同时属于多个最终角色。

        例如不能：

            users.ts

        同时属于：

            implementation_targets
            dependencies

        Investigation 阶段允许不确定，
        Resolver 阶段必须形成唯一决策。
        """

        role_groups = {
            "implementation_targets": (self.implementation_targets),
            "dependencies": (self.dependencies),
            "references": (self.references),
            "tests": (self.tests),
        }

        owner_by_path: dict[
            str,
            str,
        ] = {}

        for (
            role,
            paths,
        ) in role_groups.items():

            for path in paths:

                normalized = path.replace(
                    "\\",
                    "/",
                ).strip()

                if not normalized:
                    raise ValueError(f"{role} 中不能包含空路径")

                previous_role = owner_by_path.get(normalized)

                if previous_role is not None:
                    raise ValueError(
                        (
                            f"文件 {normalized} "
                            "同时出现在 "
                            f"{previous_role} "
                            f"和 {role}"
                        )
                    )

                owner_by_path[normalized] = role

        return self
