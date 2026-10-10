from pathlib import Path
from typing import Protocol

from agent_core.codebase.model import (
    RepositoryFileIndex,
    RepositoryProfile,
)
from agent_core.codebase.task_retrieval import (
    TaskRetrievalService,
)
from agent_core.context.builder import (
    ProjectContextBuilder,
)
from agent_core.context.model import (
    ContextBudgetConfig,
    TaskContextBundle,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


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
    ) -> ContextBudgetConfig: ...


class DefaultTaskContextFactory:
    """
    TaskContextBundle 的应用级组装器。

    职责：

    Task
      -> Retrieval
      -> Relevant File Selection
      -> Engineering Rules
      -> Context Budget
      -> ProjectContextBuilder
      -> TaskContextBundle

    不负责：
    - Exact / Semantic Retrieval 具体算法
    - 文件内容裁剪策略
    - LLM Prompt Rendering
    """

    def __init__(
        self,
        *,
        repository_root: str | Path,
        requirement: RequirementSpec,
        technical_spec: TechnicalSpec,
        repository: RepositoryProfile,
        index: RepositoryFileIndex,
        retrieval_service: TaskRetrievalService,
        context_builder: ProjectContextBuilder,
        rules_provider: EngineeringRulesProvider,
        budget_provider: ContextBudgetProvider,
    ) -> None:
        self.repository_root = Path(repository_root)

        self.requirement = requirement
        self.technical_spec = technical_spec

        self.repository = repository
        self.index = index

        self.retrieval_service = retrieval_service

        self.context_builder = context_builder

        self.rules_provider = rules_provider

        self.budget_provider = budget_provider

    async def build(
        self,
        task: TaskDefinition,
    ) -> TaskContextBundle:
        selection = await self.retrieval_service.retrieve(
            task=task,
            index=self.index,
        )

        engineering_rules = await self.rules_provider.get_rules(
            task=task,
        )

        budget = self.budget_provider.get_budget(
            task=task,
        )

        return self.context_builder.build(
            repository_root=(self.repository_root),
            requirement=(self.requirement),
            technical_spec=(self.technical_spec),
            task=task,
            repository=(self.repository),
            selection=selection,
            engineering_rules=(engineering_rules),
            max_files=(budget.max_files),
            max_total_chars=(budget.max_total_chars),
            max_chars_per_file=(budget.max_chars_per_file),
        )
