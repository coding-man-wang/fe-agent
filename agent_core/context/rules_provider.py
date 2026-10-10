from agent_core.sdd.model import (
    TaskDefinition,
)


class StaticEngineeringRulesProvider:

    def __init__(
            self,
            rules: list[str],
    ) -> None:
        self.rules = list(rules)

    async def get_rules(
            self,
            *,
            task: TaskDefinition,
    ) -> list[str]:
        return list(self.rules)
