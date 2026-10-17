from pathlib import PurePosixPath

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
    Context Role Classifier v0.1

    当前只做可解释的规则分类。

    注意：

    Role Classification 和
    Relevance Ranking 是两个不同问题。

    Reranker:
        这个文件跟任务相关吗？

    ContextRoleClassifier:
        这个文件准备拿来做什么？
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

        role, reason = self._classify_role(
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
        # 1. Test Context
        # ==============================

        if item.category == "test":

            return (
                "test",
                ("文件 category=test，" "作为测试上下文使用"),
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
        # 3. Explicit Target
        # ==============================

        target = self._find_strong_target(
            path=item.relative_path,
            targets=task.target_modules,
        )

        if target is not None:

            return (
                "implementation_target",
                ("文件与 Task target_modules " f"强匹配: {target}"),
            )

        # ==============================
        # 4. Code Change Top Evidence
        #
        # Planner 不一定能精确给出文件。
        # 对 code_change Task，
        # 高排名 implementation query
        # 可以作为 target 候选。
        # ==============================

        if (
            task.task_type == "code_change"
            and "implementation_location" in intents
            and item.rerank_rank is not None
            and item.rerank_rank <= 3
        ):

            return (
                "implementation_target",
                (
                    "code_change Task 中被 "
                    "implementation_location 命中，"
                    "且 rerank_rank <= 3"
                ),
            )

        # ==============================
        # 5. Similar Implementation
        # ==============================

        if "similar_implementation" in intents:

            return (
                "reference",
                ("由 similar_implementation " "Intent 命中"),
            )

        # ==============================
        # 6. Dependency Context
        # ==============================

        if "dependency_context" in intents:

            return (
                "dependency",
                ("由 dependency_context " "Intent 命中"),
            )

        # ==============================
        # 7. Investigation 本身就是
        #    理解依赖/现状
        # ==============================

        if task.task_type == "investigation" and item.category in {
            "source",
            "config",
        }:

            return (
                "dependency",
                ("investigation Task " "中的源码上下文"),
            )

        # ==============================
        # 8. Unknown
        # ==============================

        return (
            "unclassified",
            ("当前证据不足以确定 " "Context Role"),
        )

    @staticmethod
    def _find_strong_target(
        *,
        path: str,
        targets: list[str],
    ) -> str | None:
        """
        只接受比较明确的 target。

        精确文件:
            src/views/UsersView.vue

        Symbol:
            UsersView

        不把：
            src/views

        这种大目录里的所有文件
        都判成 implementation_target。
        """

        normalized_path = (
            path.replace(
                "\\",
                "/",
            )
            .lower()
            .strip("/")
        )

        path_name = PurePosixPath(normalized_path).name

        for target in targets:

            normalized_target = (
                target.replace(
                    "\\",
                    "/",
                )
                .lower()
                .strip("/")
            )

            if not normalized_target:
                continue

            target_path = PurePosixPath(normalized_target)

            target_name = target_path.name

            # 精确文件
            if target_path.suffix:

                if (
                    normalized_path == normalized_target
                    or normalized_path.endswith("/" + normalized_target)
                    or path_name == target_name
                ):
                    return target

                continue

            # 目录太粗，不作为 strong target
            if "/" in normalized_target:
                continue

            # Symbol / basename
            if normalized_target in path_name:
                return target

        return None
