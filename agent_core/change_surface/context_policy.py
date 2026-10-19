from dataclasses import dataclass

from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
)
from agent_core.sdd.model import TaskDefinition


@dataclass(frozen=True)
class RoleBudget:
    implementation_target: int = 4
    dependency: int = 3
    reference: int = 1
    test: int = 1


class RoleAwareContextPolicy:
    """在 Change Surface 已解析后按用途分配 Context Budget。"""

    def __init__(self, *, default_budget: RoleBudget | None = None) -> None:
        self.default_budget = default_budget or RoleBudget()

    def apply(
        self,
        *,
        selection: HybridRelevantFileSelection,
        surface: ResolvedChangeSurface,
        task: TaskDefinition,
    ) -> HybridRelevantFileSelection:
        budget = self._budget_for_task(task)
        roles = self._role_map(surface)
        buckets: dict[str, list[HybridRelevantFile]] = {
            "implementation_target": [],
            "dependency": [],
            "reference": [],
            "test": [],
        }

        for item in selection.selected_files:
            role = roles.get(item.relative_path)
            if role is None:
                continue
            buckets[role].append(
                item.model_copy(update={"context_role": role})
            )

        for items in buckets.values():
            items.sort(
                key=lambda item: (
                    item.rerank_rank or 10_000,
                    -float(item.rerank_score or 0.0),
                    item.relative_path,
                )
            )

        chosen = [
            *buckets["implementation_target"][: budget.implementation_target],
            *buckets["dependency"][: budget.dependency],
            *buckets["reference"][: budget.reference],
            *buckets["test"][: budget.test],
        ]
        chosen_paths = {item.relative_path for item in chosen}
        skipped = [
            item.relative_path
            for item in selection.selected_files
            if item.relative_path not in chosen_paths
        ]
        return selection.model_copy(
            deep=True,
            update={
                "selected_files": chosen,
                "selected_count": len(chosen),
                "total_selected_bytes": sum(item.size_bytes for item in chosen),
                "skipped_by_budget": [
                    *selection.skipped_by_budget,
                    *skipped,
                ],
            },
        )

    def _budget_for_task(self, task: TaskDefinition) -> RoleBudget:
        if task.task_type == "test":
            return RoleBudget(
                implementation_target=2,
                dependency=2,
                reference=1,
                test=4,
            )
        return self.default_budget

    @staticmethod
    def _role_map(surface: ResolvedChangeSurface) -> dict[str, str]:
        role_by_path = {
            path: "implementation_target" for path in surface.implementation_targets
        }
        role_by_path.update({path: "dependency" for path in surface.dependencies})
        role_by_path.update({path: "reference" for path in surface.references})
        role_by_path.update({path: "test" for path in surface.tests})
        return role_by_path
