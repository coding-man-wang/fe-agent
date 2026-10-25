from dataclasses import dataclass

from agent_core.change_surface.model import (
    ResolvedChangeSurface,
)
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


@dataclass(frozen=True)
class RoleBudget:
    implementation_target: int = 4
    dependency: int = 3
    reference: int = 1
    test: int = 1

    # Retrieval / Graph 找到，
    # 但没有进入 Change Surface 的文件。
    #
    # 这些文件只用于阅读，
    # 绝不扩大 Write Permission。
    context_only: int = 2


class RoleAwareContextPolicy:
    """
    在 Change Surface 已解析后，
    按用途分配 Context Budget。

    核心原则：

        Context Selection
        !=
        Change Permission

    Change Surface 决定：
        哪些文件允许修改。

    Retrieval / Graph 决定：
        哪些额外文件值得阅读。

    因此：

        不在 Change Surface 中的文件
        仍然可以作为 read-only context
        进入最终 code_context。
    """

    def __init__(
        self,
        *,
        default_budget: RoleBudget | None = None,
    ) -> None:
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

        buckets: dict[
            str,
            list[HybridRelevantFile],
        ] = {
            "implementation_target": [],
            "dependency": [],
            "reference": [],
            "test": [],
            "context_only": [],
        }

        for item in selection.selected_files:

            surface_role = roles.get(item.relative_path)

            # ==================================
            # 1. Change Surface 文件
            #
            # Change Surface 的角色拥有最高优先级。
            # ==================================

            if surface_role is not None:

                buckets[surface_role].append(
                    item.model_copy(update={"context_role": (surface_role)})
                )

                continue

            # ==================================
            # 2. Context-only 文件
            #
            # Retrieval / Graph 找到了，
            # 但它不属于 Change Surface。
            #
            # 可以读，
            # 不能因此获得写权限。
            # ==================================

            buckets["context_only"].append(item)

        # ==================================
        # Deterministic Ranking
        # ==================================

        for items in buckets.values():

            items.sort(
                key=lambda item: (
                    (item.rerank_rank or item.retrieval_rank or 10_000),
                    -float(item.rerank_score or item.selection_score or 0.0),
                    item.relative_path,
                )
            )

        # ==================================
        # Surface Context
        # ==================================

        chosen = [
            *buckets["implementation_target"][: budget.implementation_target],
            *buckets["dependency"][: budget.dependency],
            *buckets["reference"][: budget.reference],
            *buckets["test"][: budget.test],
        ]

        # ==================================
        # Read-only Context
        #
        # 注意：
        # 这里只增加阅读上下文，
        # 不修改 surface。
        # ==================================

        chosen.extend(buckets["context_only"][: budget.context_only])

        chosen_paths = {item.relative_path for item in chosen}

        skipped = [
            item.relative_path
            for item in selection.selected_files
            if (item.relative_path not in chosen_paths)
        ]

        return selection.model_copy(
            deep=True,
            update={
                "selected_files": (chosen),
                "selected_count": (len(chosen)),
                "total_selected_bytes": (sum(item.size_bytes for item in chosen)),
                "skipped_by_budget": [
                    *selection.skipped_by_budget,
                    *skipped,
                ],
            },
        )

    def _budget_for_task(
        self,
        task: TaskDefinition,
    ) -> RoleBudget:

        if task.task_type == "test":

            return RoleBudget(
                implementation_target=2,
                dependency=2,
                reference=1,
                test=4,
                context_only=2,
            )

        return self.default_budget

    @staticmethod
    def _role_map(
        surface: ResolvedChangeSurface,
    ) -> dict[str, str]:

        role_by_path = {
            path: "implementation_target" for path in surface.implementation_targets
        }

        role_by_path.update({path: "dependency" for path in surface.dependencies})

        role_by_path.update({path: "reference" for path in surface.references})

        role_by_path.update({path: "test" for path in surface.tests})

        return role_by_path
