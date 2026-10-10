from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.context.model import (
    CodeContextItem,
    RepositoryProfile,
)


class RepositoryContextResult(BaseModel):
    """
    RepositoryContextProvider 的统一输出。

    repository:
        当前仓库的整体 Profile。

    code_context:
        针对当前 Task 最终选出的代码上下文。
    """

    model_config = ConfigDict(extra="forbid")

    repository: RepositoryProfile

    code_context: list[CodeContextItem] = Field(default_factory=list)
