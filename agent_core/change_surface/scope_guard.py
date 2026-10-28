from agent_core.change_surface.errors import ScopeViolationError
from agent_core.change_surface.model import ResolvedChangeSurface, ScopeGuardResult
from agent_core.coding.patch_generation_model import FilePatchProposal
from agent_core.coding.patch_model import PatchPlan


class PatchScopeGuard:
    """在 Patch Apply 前用 ResolvedChangeSurface 做硬边界校验。"""

    def evaluate_plan(
        self,
        *,
        plan: PatchPlan,
        surface: ResolvedChangeSurface,
    ) -> ScopeGuardResult:
        requested = [
            (
                item.relative_path,
                item.action,
            )
            for item in plan.targets
        ]
        return self._evaluate(
            task_id=plan.task_id,
            requested=requested,
            surface=surface,
        )

    def evaluate_proposal(
        self,
        *,
        proposal: FilePatchProposal,
        surface: ResolvedChangeSurface,
    ) -> ScopeGuardResult:
        requested = [
            (
                item.relative_path,
                item.action,
            )
            for item in proposal.patches
        ]
        return self._evaluate(
            task_id=proposal.task_id,
            requested=requested,
            surface=surface,
        )

    def require_plan(self, *, plan: PatchPlan, surface: ResolvedChangeSurface) -> None:
        result = self.evaluate_plan(plan=plan, surface=surface)
        if not result.allowed:
            raise ScopeViolationError("; ".join(result.violations))

    def require_proposal(
        self,
        *,
        proposal: FilePatchProposal,
        surface: ResolvedChangeSurface,
    ) -> None:
        result = self.evaluate_proposal(proposal=proposal, surface=surface)
        if not result.allowed:
            raise ScopeViolationError("; ".join(result.violations))

    @staticmethod
    def _evaluate(
        *,
        task_id: str,
        requested: list[
            tuple[
                str,
                str,
            ]
        ],
        surface: ResolvedChangeSurface,
    ) -> ScopeGuardResult:

        allowed = surface.allowed_change_files()

        create_allowed = set(surface.create_files)

        violations: list[str] = []

        for path, action in requested:

            if path not in allowed:
                violations.append("ScopeViolation: " f"{path}")
                continue

            if action == "create":
                if path not in create_allowed:
                    violations.append("CreateScopeViolation: " f"{path}")

            elif action == "modify":
                if path in create_allowed:
                    violations.append("ModifyCreateTarget" "Violation: " f"{path}")

            else:
                violations.append("UnknownAction: " f"{path}:{action}")

        requested_paths = [path for path, _ in requested]

        return ScopeGuardResult(
            task_id=task_id,
            allowed=(not violations),
            requested_files=(requested_paths),
            allowed_files=sorted(allowed),
            violations=sorted(set(violations)),
        )
