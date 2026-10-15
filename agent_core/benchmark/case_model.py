from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

BenchmarkCategory = Literal[
    "single_file_change",
    "cross_file_change",
    "parallel_task",
    "bug_fix",
    "review_revision",
]


class BenchmarkExpectation(BaseModel):
    """
    Evaluator 使用的标准答案。

    注意：
    这部分不能传给 Agent。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    expected_files: list[str] = Field(
        default_factory=list,
    )

    forbidden_files: list[str] = Field(
        default_factory=list,
    )

    expected_behaviors: list[str] = Field(
        min_length=1,
    )

    verification_command_ids: list[str] = Field(
        min_length=1,
    )


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[1] = 1

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
            raise ValueError("base_commit_sha 必须是 40 位 Git SHA")

        if any(char not in "0123456789abcdef" for char in value):
            raise ValueError("base_commit_sha 必须是小写十六进制")

        return value
