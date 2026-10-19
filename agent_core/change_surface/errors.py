class ChangeSurfaceError(Exception):
    pass


class InvestigationArtifactBuildError(ChangeSurfaceError):
    pass


class ChangeSurfaceResolutionError(ChangeSurfaceError):
    pass


class ScopeViolationError(ChangeSurfaceError):
    pass
