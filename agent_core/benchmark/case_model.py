from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

BenchmarkCategory = Literal[
    "single_file_change",
    "cross_file_change",
    "parallel_task",
    "bug_fix",
    "review_revision",
]


class BenchmarkRetrievalOracle(BaseModel):
    """
    Retrieval 阶段的标准答案。

    required:
        Agent 理解任务时必须找到的文件。

    useful:
        找到有帮助，但不是必须。

    forbidden:
        不应该进入 Context 的文件。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    required_context_files: list[str] = Field(
        default_factory=list,
    )

    useful_context_files: list[str] = Field(
        default_factory=list,
    )

    forbidden_context_files: list[str] = Field(
        default_factory=list,
    )


class BenchmarkChangeOracle(BaseModel):
    """
    Coding/Patch 阶段的标准答案。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    expected_change_files: list[str] = Field(
        default_factory=list,
    )

    allowed_change_files: list[str] = Field(
        default_factory=list,
    )

    forbidden_change_files: list[str] = Field(
        default_factory=list,
    )


class BenchmarkExpectation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    # ---------------------------------
    # Schema v1
    #
    # 暂时保留，用来兼容现有 Case。
    # ---------------------------------

    expected_files: list[str] | None = None

    forbidden_files: list[str] | None = None

    # ---------------------------------
    # Schema v2
    # ---------------------------------

    required_context_files: list[str] = Field(
        default_factory=list,
    )

    useful_context_files: list[str] = Field(
        default_factory=list,
    )

    forbidden_context_files: list[str] = Field(
        default_factory=list,
    )

    expected_change_files: list[str] = Field(
        default_factory=list,
    )

    allowed_change_files: list[str] = Field(
        default_factory=list,
    )

    forbidden_change_files: list[str] = Field(
        default_factory=list,
    )

    expected_behaviors: list[str] = Field(
        min_length=1,
    )

    verification_command_ids: list[str] = Field(
        min_length=1,
    )

    # ---------------------------------
    # v1.2.4 Impact Analysis Oracle
    # ---------------------------------

    required_impact_files: list[str] = Field(
        default_factory=list,
    )

    required_impact_verification_command_ids: list[str] = Field(
        default_factory=list,
    )


class BenchmarkImpactOracle(BaseModel):
    """
    Impact Analysis 阶段的标准答案。

    required_impact_files:
        本次需求执行过程中，
        影响分析至少应该发现的文件。

        这里评价的是：
        “值得重点检查”

        不是：
        “必须修改”。

    required_verification_command_ids:
        影响分析后至少应该进入
        Verification Plan 的验证命令。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    required_impact_files: list[str] = Field(
        default_factory=list,
    )

    required_verification_command_ids: list[str] = Field(
        default_factory=list,
    )


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[
        1,
        2,
    ] = 1

    case_id: str

    title: str

    category: BenchmarkCategory

    difficulty: Literal[
        "easy",
        "medium",
        "hard",
    ]

    repository_name: str

    base_commit_sha: str

    requirement: str

    expectation: BenchmarkExpectation

    @field_validator("case_id")
    @classmethod
    def validate_case_id(
        cls,
        value: str,
    ) -> str:

        if not value.startswith("CASE-"):
            raise ValueError("case_id 必须以 CASE- 开头")

        return value

    @field_validator("base_commit_sha")
    @classmethod
    def validate_commit_sha(
        cls,
        value: str,
    ) -> str:

        if len(value) != 40:

            raise ValueError("base_commit_sha " "必须是 40 位 commit SHA")

        if value.lower() != value:

            raise ValueError("base_commit_sha " "必须使用小写")

        try:
            int(
                value,
                16,
            )

        except ValueError as exc:

            raise ValueError("base_commit_sha " "必须是十六进制") from exc

        return value

    @model_validator(mode="after")
    def validate_schema_version(
        self,
    ):

        expectation = self.expectation

        if self.schema_version == 1:

            if not (expectation.expected_files):

                raise ValueError("Schema v1 必须提供 " "expected_files")

        if self.schema_version == 2:

            if (
                expectation.expected_files is not None
                or expectation.forbidden_files is not None
            ):

                raise ValueError(
                    "Schema v2 不再使用 " "expected_files / " "forbidden_files"
                )

            if not (expectation.required_context_files):

                raise ValueError("Schema v2 必须提供 " "required_context_files")

            if not (expectation.expected_change_files):

                raise ValueError("Schema v2 必须提供 " "expected_change_files")

        return self

    def retrieval_oracle(
        self,
    ) -> BenchmarkRetrievalOracle:
        """
        无论 Case 是 v1 还是 v2，
        对 Retrieval Evaluator 都返回
        相同结构。

        这样 Evaluator 不需要知道
        Case 使用哪个 Schema。
        """

        expectation = self.expectation

        if self.schema_version == 1:

            return BenchmarkRetrievalOracle(
                required_context_files=list(expectation.expected_files or []),
                useful_context_files=[],
                # 非常重要：
                #
                # v1 forbidden_files 的真实含义
                # 更接近“不应该修改”，
                # 不应该继续拿它处罚 Retrieval。
                forbidden_context_files=[],
            )

        return BenchmarkRetrievalOracle(
            required_context_files=list(expectation.required_context_files),
            useful_context_files=list(expectation.useful_context_files),
            forbidden_context_files=list(expectation.forbidden_context_files),
        )

    def change_oracle(
        self,
    ) -> BenchmarkChangeOracle:
        """
        Patch Evaluator 后续使用。
        """

        expectation = self.expectation

        if self.schema_version == 1:

            return BenchmarkChangeOracle(
                expected_change_files=list(expectation.expected_files or []),
                allowed_change_files=[],
                forbidden_change_files=list(expectation.forbidden_files or []),
            )

        return BenchmarkChangeOracle(
            expected_change_files=list(expectation.expected_change_files),
            allowed_change_files=list(expectation.allowed_change_files),
            forbidden_change_files=list(expectation.forbidden_change_files),
        )

    def impact_oracle(
        self,
    ) -> BenchmarkImpactOracle:
        """
        v1.2.4 Impact Analysis Evaluator 使用。

        没有配置 Impact Oracle 的旧 Case，
        返回空 Oracle，不参与 Impact Gate。
        """

        expectation = self.expectation

        return BenchmarkImpactOracle(
            required_impact_files=list(expectation.required_impact_files),
            required_verification_command_ids=list(
                expectation.required_impact_verification_command_ids
            ),
        )
