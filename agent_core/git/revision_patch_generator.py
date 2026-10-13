from pathlib import Path

from agent_core.coding.patch_generator import (
    PatchGenerator,
    PatchTextModel,
)


class DefaultRevisionPatchGeneratorProvider:

    def __init__(
        self,
        *,
        model: PatchTextModel,
        max_file_bytes: int = 100_000,
    ) -> None:

        self.model = model
        self.max_file_bytes = max_file_bytes

    def create(
        self,
        *,
        repository_root: Path,
    ) -> PatchGenerator:

        return PatchGenerator(
            repository_root=(repository_root),
            model=self.model,
            max_file_bytes=(self.max_file_bytes),
        )
