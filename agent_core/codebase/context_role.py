from agent_core.codebase.model import (
    ContextRole,
    HybridRelevantFile,
    HybridRelevantFileSelection,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


class ContextRoleClassifier:
    """
    Context Role Classifier v0.2

    Reranker:
        判断文件和任务是否相关。

    Target Evidence:
        判断文件与 Planner 提供的
        Repository Hint 是否存在匹配。

        它只影响候选 Context Role，
        不授予修改权限。

    Context Role:
        判断文件最终准备在当前
        Task Context 中承担什么用途。
    """

    def classify_selection(
        self,
        *,
        selection: HybridRelevantFileSelection,
        task: TaskDefinition,
    ) -> HybridRelevantFileSelection:

        files = [
            self.classify(
                item=item,
                task=task,
            )
            for item in selection.selected_files
        ]

        return selection.model_copy(
            deep=True,
            update={
                "selected_files": files,
            },
        )

    def classify(
        self,
        *,
        item: HybridRelevantFile,
        task: TaskDefinition,
    ) -> HybridRelevantFile:

        (
            role,
            reason,
        ) = self._classify_role(
            item=item,
            task=task,
        )

        return item.model_copy(
            deep=True,
            update={
                "context_role": role,
                "role_reason": reason,
            },
        )

    def _classify_role(
        self,
        *,
        item: HybridRelevantFile,
        task: TaskDefinition,
    ) -> tuple[
        ContextRole,
        str,
    ]:

        intents = set(item.matched_intents)

        # ==============================
        # 1. Test
        #
        # 测试文件首先作为测试上下文。
        # ==============================

        if item.category == "test":

            return (
                "test",
                "文件 category=test",
            )

        # ==============================
        # 2. Project Policy
        # ==============================

        if "project_policy" in intents and item.category in {
            "config",
            "documentation",
        }:

            return (
                "policy",
                ("由 project_policy " "Intent 命中配置/文档文件"),
            )

        # ==============================
        # 3. Investigation
        #
        # Investigation 的目的不是修改，
        # 而是理解现有实现、依赖和边界。
        #
        # 因此即使存在强 Target Evidence，
        # 在 investigation Task 中也优先
        # 解释成 dependency。
        # ==============================

        if (
            task.task_type == "investigation"
            and item.category
            in {
                "source",
                "config",
            }
            and (
                "dependency_context" in intents
                or ("implementation_location" in intents)
            )
        ):

            return (
                "dependency",
                ("investigation Task 中 " "被实现/依赖查询命中的上下文"),
            )

        # ==============================
        # 4. Strong Target Evidence
        #
        # exact_file / symbol
        # 才属于明确的 Change Target。
        #
        # directory 只是区域提示，
        # 不能直接判 implementation_target。
        # ==============================

        if task.task_type == "code_change" and item.target_match_type in {
            "exact_file",
            "symbol",
        }:

            return (
                "implementation_target",
                (
                    "强 Target Evidence: "
                    f"{item.target_match_type}"
                    " -> "
                    f"{item.target_match_value}"
                ),
            )

        # ==============================
        # 5. Code Change Fallback
        #
        # 暂时保留高排名启发式，
        # 防止当前 Planner 只给目录时，
        # implementation_target 全部丢失。
        #
        # 这是 fallback，不是最终规则。
        # ==============================

        if (
            task.task_type == "code_change"
            and ("implementation_location" in intents)
            and item.rerank_rank is not None
            and item.rerank_rank <= 2
        ):

            return (
                "implementation_target",
                (
                    "fallback: code_change "
                    "Task 中 "
                    "implementation_location "
                    "高排名候选"
                ),
            )

        # ==============================
        # 6. Dependency
        # ==============================

        if "dependency_context" in intents:

            return (
                "dependency",
                ("由 dependency_context " "Intent 命中"),
            )

        # ==============================
        # 7. Similar Implementation
        #
        # 只有主要作为 similar context
        # 出现时才标 reference。
        #
        # 如果同时有 implementation /
        # dependency Evidence，
        # 不应直接降级成 reference。
        # ==============================

        if (
            "similar_implementation" in intents
            and ("implementation_location" not in intents)
            and ("dependency_context" not in intents)
        ):

            return (
                "reference",
                ("主要由 " "similar_implementation " "Intent 命中"),
            )

        # ==============================
        # 8. Unknown
        # ==============================

        return (
            "unclassified",
            ("当前 Evidence 不足以确定 " "Context Role"),
        )
