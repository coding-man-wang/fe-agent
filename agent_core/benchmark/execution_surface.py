from __future__ import annotations

from agent_core.change_surface.model import ResolvedChangeSurface


class ProductionExecutionSurfacePolicy:
    """Build the side-effect boundary for v1.1 production-only execution.

    A task-level Change Surface may contain test files because they are useful
    evidence and can be legitimate modification targets in a later test-authoring
    phase. v1.1 intentionally executes production changes only, so test files must
    remain readable but must not be advertised to the coding model as writable.

    The policy therefore converts ``tests`` into read-only ``references`` before
    Coding Context is rendered. Scope Guard then consumes the exact same surface.
    """

    def apply(self, surface: ResolvedChangeSurface) -> ResolvedChangeSurface:
        references = self._dedupe_preserving_order(
            [
                *surface.references,
                *surface.tests,
            ]
        )

        rationale = [
            *surface.rationale,
            (
                "execution-policy: v1.1 production-only 模式下，测试文件保留为"
                "只读 reference，不进入可修改 tests 边界。"
            ),
        ]

        return surface.model_copy(
            deep=True,
            update={
                "references": references,
                "tests": [],
                "rationale": rationale,
                "create_files": [],
            },
        )

    @staticmethod
    def _dedupe_preserving_order(paths: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for path in paths:
            normalized = path.replace("\\", "/").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result


class TestExecutionSurfacePolicy:
    """
    Test Task 的 Side Effect Boundary。

    test task：

    - production implementation targets 只能读取；
    - surface.tests 才允许修改；
    - dependencies / references 继续只读。

    这样测试任务不能借机修改生产代码。
    """

    def apply(
        self,
        surface: ResolvedChangeSurface,
    ) -> ResolvedChangeSurface:

        references = self._dedupe_preserving_order(
            [
                *surface.references,
                *surface.dependencies,
                *surface.implementation_targets,
            ]
        )

        rationale = [
            *surface.rationale,
            (
                "execution-policy: test task "
                "只允许修改测试文件；"
                "production targets "
                "保留为只读 reference。"
            ),
        ]

        return surface.model_copy(
            deep=True,
            update={
                "implementation_targets": [],
                "dependencies": [],
                "references": references,
                # 保留测试文件可写。
                "tests": list(surface.tests),
                "rationale": rationale,
                "create_files": list(surface.create_files),
            },
        )

    @staticmethod
    def _dedupe_preserving_order(
        paths: list[str],
    ) -> list[str]:

        seen: set[str] = set()

        result: list[str] = []

        for path in paths:

            normalized = path.replace("\\", "/").strip()

            if not normalized or normalized in seen:
                continue

            seen.add(normalized)

            result.append(normalized)

        return result
