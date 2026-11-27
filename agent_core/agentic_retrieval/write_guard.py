"""Persisted read-only provenance must survive downstream Surface resolution."""
from __future__ import annotations

from agent_core.change_surface.errors import ChangeSurfaceResolutionError


def _key(path: str) -> str:
    # Conservative on both Windows and Linux; Surface normally uses canonical
    # relative paths already. Do not allow case/./ spelling to bypass the guard.
    path = path.replace("\\", "/").strip()
    while path.startswith("./"):
        path = path[2:]
    return path.casefold()


def assert_read_only_paths(surface, investigations) -> None:
    denied = {_key(path) for artifact in investigations
              for path in artifact.read_only_paths}
    if not denied:
        return
    proposed = [*surface.implementation_targets, *surface.tests, *surface.create_files]
    violations = sorted({path for path in proposed if _key(path) in denied})
    if violations:
        raise ChangeSurfaceResolutionError(
            "v1.6 read-only evidence cannot become a write target without prior host approval: "
            + ", ".join(violations))
