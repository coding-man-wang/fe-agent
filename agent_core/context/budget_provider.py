from agent_core.context.model import (
    ContextBudgetConfig,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


class StaticContextBudgetProvider:

    def __init__(
        self,
        budget: ContextBudgetConfig,
    ) -> None:
        self.budget = budget

    def get_budget(
        self,
        *,
        task: TaskDefinition,
    ) -> ContextBudgetConfig:
        return self.budget
