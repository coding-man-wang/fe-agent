from typing import Protocol

from agent_core.context.repository_provider_model import (
    RepositoryContextResult,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


class TaskCodeRetriever(Protocol):
    def retrieve(
            self,
            *,
            task: TaskDefinition,
    ) -> list: ...


class TaskRelevantFileSelector(Protocol):
    def select(
            self,
            *,
            task: TaskDefinition,
            candidates: list,
    ) -> list: ...


class DefaultRepositoryContextProvider:

    def __init__(
            self,
            *,
            repository_profile,
            retriever: TaskCodeRetriever,
            selector: TaskRelevantFileSelector,
    ) -> None:
        self.repository_profile = repository_profile

        self.retriever = retriever
        self.selector = selector

    async def build_repository_context(
            self,
            *,
            task: TaskDefinition,
    ) -> RepositoryContextResult:
        candidates = self.retriever.retrieve(task=task)

        selected = self.selector.select(
            task=task,
            candidates=candidates,
        )

        return RepositoryContextResult(
            repository=(self.repository_profile),
            code_context=selected,
        )
