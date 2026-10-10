from typing import Protocol

from agent_core.context.model import (
    ContextBudgetUsage,
    TaskContextBundle,
)
from agent_core.context.repository_provider_model import (
    RepositoryContextResult,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


class RepositoryContextProvider(Protocol):
    async def build_repository_context(
            self,
            *,
            task: TaskDefinition,
    ) -> RepositoryContextResult: ...


class EngineeringRulesProvider(Protocol):
    async def get_rules(
            self,
            *,
            task: TaskDefinition,
    ) -> list[str]: ...


class ContextBudgetProvider(Protocol):
    def get_budget(
            self,
            *,
            task: TaskDefinition,
    ) -> ContextBudgetUsage: ...


class DefaultTaskContextFactory:

    def __init__(
            self,
            *,
            requirement: RequirementSpec,
            technical_spec: TechnicalSpec,
            repository_provider: RepositoryContextProvider,
            rules_provider: EngineeringRulesProvider,
            budget_provider: ContextBudgetProvider,
    ) -> None:
        self.requirement = requirement
        self.technical_spec = technical_spec
        self.repository_provider = repository_provider
        self.rules_provider = rules_provider
        self.budget_provider = budget_provider

    async def build(
            self,
            task: TaskDefinition,
    ) -> TaskContextBundle:
        repository_result = await self.repository_provider.build_repository_context(
            task=task
        )

        engineering_rules = await self.rules_provider.get_rules(task=task)

        budget = self.budget_provider.get_budget(task=task)

        return TaskContextBundle(
            requirement=(self.requirement),
            technical_spec=(self.technical_spec),
            task=task,
            repository=(repository_result.repository),
            engineering_rules=(engineering_rules),
            code_context=(repository_result.code_context),
            budget=budget,
        )
