from pathlib import Path

import pytest

from agent_core.coding.approval import (
    ApprovalError,
)
from agent_core.coding.approval_store import (
    ApprovalStore,
)
from agent_core.coding.patch_apply import (
    PatchApplyError,
    PatchApplyService,
)
from agent_core.coding.reviewer import (
    PatchReviewer,
)
from tests.test_patch_review import (
    BEFORE,
    AFTER,
    make_proposal,
)


def create_environment(
    root: Path,
    *,
    approve: bool,
):
    proposal = make_proposal()

    target = root / "src/api/user.ts"

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_bytes(BEFORE.encode("utf-8"))

    store = ApprovalStore(root)

    review = PatchReviewer().prepare(proposal)

    if approve:
        store.record_decision(
            review=review,
            decision="approved",
        )

    service = PatchApplyService(
        repository_root=root,
        approval_store=store,
    )

    return (
        proposal,
        store,
        service,
        target,
    )


def test_approved_patch_writes_file(
    tmp_path: Path,
):
    proposal, _, service, target = create_environment(
        tmp_path,
        approve=True,
    )

    result = service.apply(proposal=proposal)

    assert target.read_text(encoding="utf-8") == AFTER

    assert result.status == "file_written"

    assert result.verification_status == "not_run"


def test_unapproved_patch_does_not_write(
    tmp_path: Path,
):
    proposal, _, service, target = create_environment(
        tmp_path,
        approve=False,
    )

    with pytest.raises(
        ApprovalError,
        match="找不到审批记录",
    ):
        service.apply(proposal=proposal)

    assert target.read_text(encoding="utf-8") == BEFORE


def test_stale_file_does_not_write(
    tmp_path: Path,
):
    proposal, _, service, target = create_environment(
        tmp_path,
        approve=True,
    )

    changed = "export function changed() {}\n"

    target.write_text(
        changed,
        encoding="utf-8",
    )

    with pytest.raises(
        PatchApplyError,
        match="版本已经变化",
    ):
        service.apply(proposal=proposal)

    assert target.read_text(encoding="utf-8") == changed


def test_reapplying_same_patch_is_rejected(
    tmp_path: Path,
):
    proposal, _, service, target = create_environment(
        tmp_path,
        approve=True,
    )

    service.apply(proposal=proposal)

    with pytest.raises(
        PatchApplyError,
        match="版本已经变化",
    ):
        service.apply(proposal=proposal)

    assert target.read_text(encoding="utf-8") == AFTER


def test_multi_file_patch_is_rejected(
    tmp_path: Path,
):
    proposal, store, service, target = create_environment(
        tmp_path,
        approve=False,
    )

    second_patch = proposal.patches[0].model_copy(
        deep=True,
        update={"relative_path": ("src/api/another.ts")},
    )

    # 当前测试只验证多文件执行边界。
    # 第二份 Patch 的完整 Diff 需要同步生成。
    import difflib

    second_patch.unified_diff = "".join(
        difflib.unified_diff(
            second_patch.before_content.splitlines(keepends=True),
            second_patch.after_content.splitlines(keepends=True),
            fromfile="a/src/api/another.ts",
            tofile="b/src/api/another.ts",
        )
    )

    proposal.patches.append(second_patch)

    review = PatchReviewer().prepare(proposal)

    store.record_decision(
        review=review,
        decision="approved",
    )

    with pytest.raises(
        PatchApplyError,
        match="只支持单文件",
    ):
        service.apply(proposal=proposal)

    assert target.read_text(encoding="utf-8") == BEFORE
