from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
)
from agent_core.git.change_set_model import (
    ChangeSetValidationResult,
    GitChangeSet,
)


class ChangeSetValidator:

    def validate(
        self,
        *,
        proposal: FilePatchProposal,
        change_set: GitChangeSet,
    ) -> ChangeSetValidationResult:

        expected_paths = {patch.relative_path for patch in proposal.patches}

        actual_paths = change_set.paths

        unexpected_paths = sorted(actual_paths - expected_paths)

        missing_paths = sorted(expected_paths - actual_paths)

        invalid_operations: list[str] = []

        for change in change_set.files:

            # 当前 PatchGenerator v0.1
            # 只支持 modify。
            #
            # 所以 Git 最终也只能看到
            # modified。
            if change.kind != "modified":

                invalid_operations.append((f"{change.relative_path}:" f"{change.kind}"))

        allowed = not unexpected_paths and not missing_paths and not invalid_operations

        return ChangeSetValidationResult(
            allowed=allowed,
            expected_paths=sorted(expected_paths),
            actual_paths=sorted(actual_paths),
            unexpected_paths=(unexpected_paths),
            missing_paths=(missing_paths),
            invalid_operations=(sorted(invalid_operations)),
        )
