from __future__ import annotations

from typing import TYPE_CHECKING

from collections import defaultdict

from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.change_surface.unresolved import UnresolvedQuestionConsolidator
if TYPE_CHECKING:
    from agent_core.sdd.model import TaskPlan


class ChangeSurfaceAggregator:
    """
    把多个 Task 级 ResolvedChangeSurface 合并成 Case/Requirement 级 Surface。

    与旧版不同：
    - 不再简单使用 implementation > test > dependency > reference 的全局优先级；
    - 会结合 Task Type 和文件类型判断最终角色；
    - test Task 不能把生产代码升级为 Case 级 implementation target；
    - 测试文件即使某个 Task 误标为 implementation target，Case 级也归入 tests；
    - unresolved 会做确定性近似去重和数量收敛。
    """

    ROLE_ORDER = (
        "implementation_targets",
        "tests",
        "dependencies",
        "references",
    )

    @classmethod
    def combine(
        cls,
        *,
        task_id: str,
        surfaces: list[ResolvedChangeSurface],
        task_plan: TaskPlan | None = None,
        unresolved_consolidator: UnresolvedQuestionConsolidator | None = None,
    ) -> ResolvedChangeSurface:
        if not task_id.strip():
            raise ValueError("task_id 不能为空")

        task_type_by_id = {
            task.task_id: task.task_type
            for task in (task_plan.tasks if task_plan is not None else [])
        }

        appearances: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        original_path_by_normalized: dict[str, str] = {}
        unresolved: list[str] = []
        rationale: list[str] = []

        for surface in surfaces:
            task_type = task_type_by_id.get(surface.task_id, "unknown")
            unresolved.extend(surface.unresolved)
            rationale.extend(surface.rationale)

            for role in cls.ROLE_ORDER:
                for raw_path in getattr(surface, role):
                    normalized = cls._normalize_path(raw_path)
                    if not normalized:
                        continue
                    appearances[normalized].append(
                        (surface.task_id, task_type, role)
                    )
                    original_path_by_normalized.setdefault(normalized, raw_path)

        merged: dict[str, list[str]] = {
            role: [] for role in cls.ROLE_ORDER
        }

        for normalized_path in sorted(appearances):
            entries = appearances[normalized_path]
            original_path = original_path_by_normalized[normalized_path]
            final_role = cls._resolve_case_role(
                path=original_path,
                entries=entries,
            )
            merged[final_role].append(original_path)

            roles = sorted({entry[2] for entry in entries})
            task_types = sorted({entry[1] for entry in entries})
            if len(roles) > 1 or len(task_types) > 1:
                rationale.append(
                    "aggregation: "
                    f"{original_path} 在 Task 中出现 roles={roles}, "
                    f"task_types={task_types}，按 Task-aware 规则归一化为 {final_role}"
                )

        consolidator = unresolved_consolidator or UnresolvedQuestionConsolidator()
        consolidated_unresolved = consolidator.consolidate(unresolved)

        implementation_targets = sorted(merged["implementation_targets"])
        tests = sorted(merged["tests"])
        if implementation_targets or tests:
            disposition = "change_required"
        elif consolidated_unresolved:
            disposition = "insufficient_evidence"
        else:
            disposition = "no_change_required"

        return ResolvedChangeSurface(
            task_id=task_id,
            implementation_targets=implementation_targets,
            dependencies=sorted(merged["dependencies"]),
            references=sorted(merged["references"]),
            tests=tests,
            unresolved=consolidated_unresolved,
            rationale=rationale,
            execution_disposition=disposition,
        )

    @classmethod
    def _resolve_case_role(
        cls,
        *,
        path: str,
        entries: list[tuple[str, str, str]],
    ) -> str:
        if cls._is_test_path(path):
            # 测试文件的“可修改角色”永远是 tests，而不是生产 implementation target。
            if any(role in {"implementation_targets", "tests"} for _, _, role in entries):
                return "tests"
            if any(role == "dependencies" for _, _, role in entries):
                return "dependencies"
            return "references"

        # 只有 code_change Task 对生产文件的 implementation_target 判断
        # 才能提升为 Case 级 implementation target。
        if any(
            task_type in {"code_change", "unknown"} and role == "implementation_targets"
            for _, task_type, role in entries
        ):
            return "implementation_targets"

        if any(
            task_type == "test" and role == "implementation_targets"
            for _, task_type, role in entries
        ):
            return "dependencies"

        if any(role == "dependencies" for _, _, role in entries):
            return "dependencies"

        return "references"

    @staticmethod
    def _normalize_path(path: str) -> str:
        return path.replace("\\", "/").strip()

    @staticmethod
    def _is_test_path(path: str) -> bool:
        normalized = path.replace("\\", "/").casefold()
        return (
            normalized.startswith("tests/")
            or "/tests/" in normalized
            or normalized.endswith(".spec.ts")
            or normalized.endswith(".test.ts")
            or normalized.endswith(".spec.tsx")
            or normalized.endswith(".test.tsx")
            or normalized.endswith(".spec.js")
            or normalized.endswith(".test.js")
        )
