from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "ArtifactAwareRetrievalQueryBuilder",
    "ArtifactDagBuilder",
    "ArtifactGraph",
    "ArtifactGraphEdge",
    "ArtifactGraphNode",
    "ArtifactType",
    "TaskArtifact",
    "TaskArtifactResolver",
    "JsonTaskArtifactStore",
]

_LAZY_IMPORTS = {
    "ArtifactAwareRetrievalQueryBuilder": (
        "agent_core.artifacts.query_builder",
        "ArtifactAwareRetrievalQueryBuilder",
    ),
    "ArtifactDagBuilder": (
        "agent_core.artifacts.graph",
        "ArtifactDagBuilder",
    ),
    "ArtifactGraph": (
        "agent_core.artifacts.model",
        "ArtifactGraph",
    ),
    "ArtifactGraphEdge": (
        "agent_core.artifacts.model",
        "ArtifactGraphEdge",
    ),
    "ArtifactGraphNode": (
        "agent_core.artifacts.model",
        "ArtifactGraphNode",
    ),
    "ArtifactType": (
        "agent_core.artifacts.model",
        "ArtifactType",
    ),
    "TaskArtifact": (
        "agent_core.artifacts.model",
        "TaskArtifact",
    ),
    "TaskArtifactResolver": (
        "agent_core.artifacts.resolver",
        "TaskArtifactResolver",
    ),
    "JsonTaskArtifactStore": (
        "agent_core.artifacts.store",
        "JsonTaskArtifactStore",
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
    from agent_core.artifacts.graph import ArtifactDagBuilder
    from agent_core.artifacts.model import (
        ArtifactGraph,
        ArtifactGraphEdge,
        ArtifactGraphNode,
        ArtifactType,
        TaskArtifact,
    )
    from agent_core.artifacts.query_builder import ArtifactAwareRetrievalQueryBuilder
    from agent_core.artifacts.resolver import TaskArtifactResolver
    from agent_core.artifacts.store import JsonTaskArtifactStore
