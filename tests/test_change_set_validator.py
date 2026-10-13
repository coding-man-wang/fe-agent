from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
    ValidatedFilePatch,
)
from agent_core.git.change_set_model import (
    GitChangeSet,
    GitFileChange,
)
from agent_core.git.change_set_validator import (
    ChangeSetValidator,
)


def make_patch(
    path: str,
) -> ValidatedFilePatch:

    return ValidatedFilePatch(
        relative_path=path,
        expected_sha256="a" * 64,
        after_sha256="b" * 64,
        before_content="before",
        after_content="after",
        explanation="test",
        unified_diff="diff",
    )


def make_proposal(
    *paths: str,
) -> FilePatchProposal:

    return FilePatchProposal(
        task_id="TASK-001",
        artifact_id="a" * 64,
        context_hash="b" * 64,
        patches=[make_patch(path) for path in paths],
    )


def test_change_set_matches_proposal():
    validator = ChangeSetValidator()

    proposal = make_proposal("src/user.ts")

    change_set = GitChangeSet(
        files=[
            GitFileChange(
                relative_path=("src/user.ts"),
                kind="modified",
                raw_status=" M",
            )
        ]
    )

    result = validator.validate(
        proposal=proposal,
        change_set=change_set,
    )

    assert result.allowed is True

    assert result.unexpected_paths == []

    assert result.missing_paths == []


def test_change_set_rejects_unexpected_file():
    validator = ChangeSetValidator()

    proposal = make_proposal("src/user.ts")

    change_set = GitChangeSet(
        files=[
            GitFileChange(
                relative_path=("src/user.ts"),
                kind="modified",
                raw_status=" M",
            ),
            GitFileChange(
                relative_path=".env",
                kind="untracked",
                raw_status="??",
            ),
        ]
    )

    result = validator.validate(
        proposal=proposal,
        change_set=change_set,
    )

    assert result.allowed is False

    assert result.unexpected_paths == [".env"]


def test_change_set_rejects_missing_file():
    validator = ChangeSetValidator()

    proposal = make_proposal(
        "src/user.ts",
        "src/user-api.ts",
    )

    change_set = GitChangeSet(
        files=[
            GitFileChange(
                relative_path=("src/user.ts"),
                kind="modified",
                raw_status=" M",
            )
        ]
    )

    result = validator.validate(
        proposal=proposal,
        change_set=change_set,
    )

    assert result.allowed is False

    assert result.missing_paths == ["src/user-api.ts"]


def test_change_set_rejects_delete():
    validator = ChangeSetValidator()

    proposal = make_proposal("src/user.ts")

    change_set = GitChangeSet(
        files=[
            GitFileChange(
                relative_path=("src/user.ts"),
                kind="deleted",
                raw_status=" D",
            )
        ]
    )

    result = validator.validate(
        proposal=proposal,
        change_set=change_set,
    )

    assert result.allowed is False

    assert result.invalid_operations == ["src/user.ts:deleted"]
