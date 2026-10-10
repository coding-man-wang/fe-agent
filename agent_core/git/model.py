from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class GitWorkingTreeStatus(BaseModel):
    """
    当前 Git Working Tree 的状态。

    tracked_changes:
        已跟踪文件的修改/删除/重命名等。

    untracked_files:
        Git 尚未跟踪的新文件。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    tracked_changes: list[str] = Field(
        default_factory=list,
    )

    untracked_files: list[str] = Field(
        default_factory=list,
    )

    @property
    def is_clean(self) -> bool:
        return not self.tracked_changes and not self.untracked_files


class GitRepositoryState(BaseModel):
    """
    Agent 在开始 Workspace Isolation 前
    需要确认的仓库事实。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    repository_root: str

    head_sha: str = Field(
        min_length=40,
        max_length=40,
    )

    branch_name: str | None = None

    is_detached_head: bool

    working_tree: GitWorkingTreeStatus
