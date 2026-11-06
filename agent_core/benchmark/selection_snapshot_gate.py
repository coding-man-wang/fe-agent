from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agent_core.benchmark.selection_evidence_binding import (
    capture_selection_evidence,
)
from agent_core.codebase.model import (
    HybridRelevantFileSelection,
)


@dataclass(frozen=True)
class SnapshotGateDecision:
    allowed: bool
    reason: str


class ShadowSnapshotGate:
    """
    v1.4.9-P2-A

    验证一份既有 Shadow Proposal
    是否仍与当前任务和源码一致。

    不修改 Selection。
    不读取 Benchmark Oracle。
    不负责启用 Active Policy。
    """

    def validate(
        self,
        *,
        report: Mapping[str, Any],
        repository_root: str | Path,
        selection: HybridRelevantFileSelection,
        task_id: str,
        raw_requirement: str,
        expected_revision: str,
        current_revision: str,
    ) -> SnapshotGateDecision:

        def deny(reason: str) -> SnapshotGateDecision:
            return SnapshotGateDecision(
                allowed=False,
                reason=reason,
            )

        # 1. 必须是已完成且未应用的 P1 报告。
        if report.get("policy_version") != "v1.4.9-P1":
            return deny("unsupported_policy_version")

        if report.get("mode") != "shadow":
            return deny("invalid_mode")

        if report.get("status") != "complete":
            return deny("shadow_not_complete")

        if report.get("applied") is not False:
            return deny("proposal_already_applied")

        if report.get("selection_unchanged") is not True:
            return deny("selection_mutated")

        # 2. 验证任务和需求。
        if report.get("task_id") != task_id:
            return deny("task_mismatch")

        requirement_hash = hashlib.sha256(raw_requirement.encode("utf-8")).hexdigest()

        if report.get("requirement_sha256") != requirement_hash:
            return deny("requirement_changed")

        # 3. 调用方必须重新读取当前 Git HEAD。
        if (
            not expected_revision
            or not current_revision
            or report.get("repository_revision") != expected_revision
            or expected_revision != current_revision
        ):
            return deny("repository_revision_changed")

        # 4. 检查 P1 保存的指纹。
        binding = report.get("evidence_binding")

        if not isinstance(binding, dict):
            return deny("missing_evidence_binding")

        if (
            binding.get("source_matched") is not True
            or binding.get("selection_matched") is not True
            or binding.get("changed_paths") != []
        ):
            return deny("invalid_shadow_evidence")

        before_hash = binding.get("source_sha256_before")
        after_hash = binding.get("source_sha256_after")

        if (
            not isinstance(before_hash, str)
            or not before_hash
            or before_hash != after_hash
        ):
            return deny("invalid_source_hash")

        # 5. 校验 Proposal 文件集合的基本一致性。
        drops = report.get("effective_proposed_drops")

        if not isinstance(drops, list):
            return deny("invalid_proposed_drops")

        if not all(isinstance(path, str) and bool(path.strip()) for path in drops):
            return deny("invalid_proposed_drops")

        selected_paths = {item.relative_path for item in selection.selected_files}

        if len(drops) != len(set(drops)) or not set(drops).issubset(selected_paths):
            return deny("proposal_path_mismatch")

        # Proposal 声明的输入、输出文件数量必须可信。
        baseline_count = len(selection.selected_files)

        if report.get("baseline_file_count") != baseline_count or report.get(
            "proposed_file_count"
        ) != baseline_count - len(drops):
            return deny("proposal_count_mismatch")

        # 6. 重新读取磁盘文件。
        try:
            current = capture_selection_evidence(
                repository_root=repository_root,
                selection=selection,
            )
        except (
            OSError,
            ValueError,
            UnicodeError,
        ):
            return deny("source_snapshot_unavailable")

        # 7. 对照当时保存的快照。
        if current.selection_sha256 != binding.get("selection_sha256"):
            return deny("selection_snapshot_changed")

        if current.source_sha256 != after_hash:
            return deny("source_snapshot_changed")

        return SnapshotGateDecision(
            allowed=True,
            reason="snapshot_valid",
        )
