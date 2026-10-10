from pathlib import Path

import pytest

from agent_core.coding.snapshot import (
    FileSnapshotBuilder,
    FileSnapshotValidator,
    SnapshotError,
)
from tests.test_context_renderer import (
    create_test_bundle,
)


def prepare_repository(
    root: Path,
):
    bundle = create_test_bundle(root)

    for item in bundle.code_context:
        path = root / item.relative_path

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            item.content,
            encoding="utf-8",
        )

    return bundle


def test_build_snapshot(
    tmp_path: Path,
):
    bundle = prepare_repository(tmp_path)

    snapshot = FileSnapshotBuilder(tmp_path).build(bundle)

    assert snapshot.task_id == "TASK-003"

    assert len(snapshot.files) == 1

    item = snapshot.files[0]

    assert item.relative_path == "src/api/user.ts"

    assert len(item.sha256) == 64


def test_snapshot_validation_passes(
    tmp_path: Path,
):
    bundle = prepare_repository(tmp_path)

    snapshot = FileSnapshotBuilder(tmp_path).build(bundle)

    FileSnapshotValidator(tmp_path).validate(snapshot)


def test_snapshot_detects_file_change(
    tmp_path: Path,
):
    bundle = prepare_repository(tmp_path)

    snapshot = FileSnapshotBuilder(tmp_path).build(bundle)

    path = tmp_path / "src/api/user.ts"

    path.write_text(
        "export function changed() {}",
        encoding="utf-8",
    )

    with pytest.raises(
        SnapshotError,
        match="发生变化",
    ):
        FileSnapshotValidator(tmp_path).validate(snapshot)
