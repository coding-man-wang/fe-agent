from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class ExecutedFileChange(BaseModel):
    """
    一个已经真实写入仓库的文件变更。

    注意：
    这不是“计划修改什么”，
    而是“Runtime 已经确认修改成功什么”。
    """

    model_config = ConfigDict(extra="forbid")

    relative_path: str = Field(min_length=1)

    before_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    after_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    status: Literal["applied"] = "applied"


class ExecutionVerificationCheck(BaseModel):
    """
    Execution Artifact 中保存的验证结果摘要。

    这里只记录最终已经通过的检查，
    例如：

    - lint
    - typecheck
    - unit test
    - build
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)

    status: Literal["passed"] = "passed"

    details: str = ""


class TaskExecutionArtifact(BaseModel):
    """
    某个 Task 已经真实执行完成的跨 Task Artifact。

    它代表的不是：

        “准备修改”

    而是：

        “修改已经应用，并且验证已经通过”。

    下游 Task 可以使用它确认：

    - 哪个 Task 已完成；
    - 哪些文件真的发生了变化；
    - 修改后的文件 Hash；
    - Verification 是否通过。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1

    # upstream_artifacts 是通用 dict，
    # 因此在 payload 内也保留类型，
    # 方便消费者区分 Artifact。
    artifact_type: Literal["execution"] = "execution"

    task_id: str = Field(min_length=1)

    proposal_id: str = Field(
        min_length=64,
        max_length=64,
    )

    status: Literal["completed"] = "completed"

    execution_status: Literal["applied"] = "applied"

    verification_status: Literal["passed"] = "passed"

    repository_state: Literal["verified_after_apply"] = "verified_after_apply"

    changed_files: list[ExecutedFileChange] = Field(min_length=1)

    verification_checks: list[ExecutionVerificationCheck] = Field(default_factory=list)
