from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "ResolvedTaskContextBuilder",
    "ChangeSurfaceAggregator",
    "UnresolvedQuestionConsolidator",
    "ChangeSurfaceError",
    "ChangeSurfaceResolutionError",
    "InvestigationArtifactBuildError",
    "ScopeViolationError",
    "InvestigationArtifactBuilder",
    "ChangeSurfaceResolver",
    "RoleAwareContextPolicy",
    "RoleBudget",
    "PatchScopeGuard",
    "RepositoryAwareReplanner",
    "InvestigationArtifact",
    "ObservedFileRole",
    "ObservedRepositoryFile",
    "ResolvedChangeSurface",
    "ExecutionDisposition",
    "ScopeGuardResult",
    "RepositoryAwareTask",
    "RepositoryAwareTaskPlan",
]

_LAZY_IMPORTS = {
    "ChangeSurfaceAggregator": (
        "agent_core.change_surface.aggregate",
        "ChangeSurfaceAggregator",
    ),
    "UnresolvedQuestionConsolidator": (
        "agent_core.change_surface.unresolved",
        "UnresolvedQuestionConsolidator",
    ),
    "ResolvedTaskContextBuilder": (
        "agent_core.change_surface.context_builder",
        "ResolvedTaskContextBuilder",
    ),
    "RoleAwareContextPolicy": (
        "agent_core.change_surface.context_policy",
        "RoleAwareContextPolicy",
    ),
    "RoleBudget": (
        "agent_core.change_surface.context_policy",
        "RoleBudget",
    ),
    "ChangeSurfaceError": (
        "agent_core.change_surface.errors",
        "ChangeSurfaceError",
    ),
    "ChangeSurfaceResolutionError": (
        "agent_core.change_surface.errors",
        "ChangeSurfaceResolutionError",
    ),
    "InvestigationArtifactBuildError": (
        "agent_core.change_surface.errors",
        "InvestigationArtifactBuildError",
    ),
    "ScopeViolationError": (
        "agent_core.change_surface.errors",
        "ScopeViolationError",
    ),
    "InvestigationArtifactBuilder": (
        "agent_core.change_surface.investigation",
        "InvestigationArtifactBuilder",
    ),
    "RepositoryAwareReplanner": (
        "agent_core.change_surface.replan",
        "RepositoryAwareReplanner",
    ),
    "ChangeSurfaceResolver": (
        "agent_core.change_surface.resolver",
        "ChangeSurfaceResolver",
    ),
    "PatchScopeGuard": (
        "agent_core.change_surface.scope_guard",
        "PatchScopeGuard",
    ),
    "InvestigationArtifact": (
        "agent_core.change_surface.model",
        "InvestigationArtifact",
    ),
    "ObservedFileRole": (
        "agent_core.change_surface.model",
        "ObservedFileRole",
    ),
    "ObservedRepositoryFile": (
        "agent_core.change_surface.model",
        "ObservedRepositoryFile",
    ),
    "ResolvedChangeSurface": (
        "agent_core.change_surface.model",
        "ResolvedChangeSurface",
    ),
    "ExecutionDisposition": (
        "agent_core.change_surface.model",
        "ExecutionDisposition",
    ),
    "ScopeGuardResult": (
        "agent_core.change_surface.model",
        "ScopeGuardResult",
    ),
    "RepositoryAwareTask": (
        "agent_core.change_surface.model",
        "RepositoryAwareTask",
    ),
    "RepositoryAwareTaskPlan": (
        "agent_core.change_surface.model",
        "RepositoryAwareTaskPlan",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from agent_core.change_surface.aggregate import ChangeSurfaceAggregator
    from agent_core.change_surface.context_builder import ResolvedTaskContextBuilder
    from agent_core.change_surface.context_policy import RoleAwareContextPolicy, RoleBudget
    from agent_core.change_surface.errors import (
        ChangeSurfaceError,
        ChangeSurfaceResolutionError,
        InvestigationArtifactBuildError,
        ScopeViolationError,
    )
    from agent_core.change_surface.investigation import InvestigationArtifactBuilder
    from agent_core.change_surface.model import (
        ExecutionDisposition,
        InvestigationArtifact,
        ObservedFileRole,
        ObservedRepositoryFile,
        RepositoryAwareTask,
        RepositoryAwareTaskPlan,
        ResolvedChangeSurface,
        ScopeGuardResult,
    )
    from agent_core.change_surface.replan import RepositoryAwareReplanner
    from agent_core.change_surface.resolver import ChangeSurfaceResolver
    from agent_core.change_surface.scope_guard import PatchScopeGuard
    from agent_core.change_surface.unresolved import UnresolvedQuestionConsolidator
