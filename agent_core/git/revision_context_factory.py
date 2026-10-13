from agent_core.codebase.task_retrieval import (
    TaskRetrievalService,
)
from agent_core.context.builder import (
    ProjectContextBuilder,
)
from agent_core.context.factory import (
    ContextBudgetProvider,
    DefaultTaskContextFactory,
    EngineeringRulesProvider,
)
from agent_core.git.revision_code_snapshot import (
    RevisionCodeSnapshot,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TechnicalSpec,
)


class DefaultRevisionContextFactoryProvider:

    def __init__(
        self,
        *,
        requirement: RequirementSpec,
        technical_spec: TechnicalSpec,
        retrieval_service: TaskRetrievalService,
        context_builder: ProjectContextBuilder,
        rules_provider: EngineeringRulesProvider,
        budget_provider: ContextBudgetProvider,
    ) -> None:

        self.requirement = requirement
        self.technical_spec = technical_spec
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.rules_provider = rules_provider
        self.budget_provider = budget_provider

    def create(
        self,
        *,
        code_snapshot: RevisionCodeSnapshot,
    ) -> DefaultTaskContextFactory:

        return DefaultTaskContextFactory(
            repository_root=(code_snapshot.snapshot.snapshot_path),
            requirement=(self.requirement),
            technical_spec=(self.technical_spec),
            repository=(code_snapshot.repository),
            index=(code_snapshot.index),
            retrieval_service=(self.retrieval_service),
            context_builder=(self.context_builder),
            rules_provider=(self.rules_provider),
            budget_provider=(self.budget_provider),
        )
