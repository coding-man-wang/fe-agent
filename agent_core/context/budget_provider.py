from agent_core.context.model import (
    ContextBudgetUsage,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


class StaticContextBudgetProvider:

    def __init__(
        self,
        budget: ContextBudgetUsage,
    ) -> None:
        self.budget = budget

    def get_budget(
        self,
        *,
        task: TaskDefinition,
    ) -> ContextBudgetUsage:
        return self.budget
