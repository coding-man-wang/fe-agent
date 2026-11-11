from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4

from agent_core.codebase.model import (
    HybridRelevantFileSelection,
)


def selection_source_digest(
    *,
    repository_root: str | Path,
    selection: HybridRelevantFileSelection,
) -> str:
    """
    根据文件路径和实际源码计算稳定指纹。

    不依赖 Git Commit。
    不信任 Selection 内的源码摘要。
    不在报告中保存源码内容。
    """
    root = Path(repository_root).resolve(strict=True)

    if not root.is_dir():
        raise ValueError("repository_root 不是目录")

    entries: list[tuple[str, str]] = []
    seen: set[str] = set()

    for item in selection.selected_files:
        normalized = item.relative_path.replace("\\", "/")
        relative = PurePosixPath(normalized)

        if (
            relative.is_absolute()
            or PureWindowsPath(item.relative_path).drive
            or ".." in relative.parts
            or not relative.parts
        ):
            raise ValueError("非法 Repository Path")

        if normalized in seen:
            raise ValueError("Selection 存在重复路径")

        seen.add(normalized)

        source = (root / Path(*relative.parts)).resolve(strict=True)

        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError("源码路径非法或不可读取")

        digest = hashlib.sha256()

        with source.open("rb") as stream:
            for chunk in iter(
                lambda: stream.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        entries.append(
            (
                normalized,
                digest.hexdigest(),
            )
        )

    # 按路径排序，使指纹与文件排列顺序无关。
    payload = json.dumps(
        sorted(entries),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JsonlShadowAuditStore:
    """
    v1.4.9-P1 Shadow Audit Store.

    审计字段使用白名单：
    - 保留策略版本和执行结果
    - 保留源码及 Selection 指纹
    - 保留删除建议及异常原因
    - 不持久化原始 Prompt、源码和 API Key
    """

    SIMPLE_FIELDS = (
        "policy_version",
        "mode",
        "task_id",
        "repository_revision",
        "requirement_sha256",
        "applied",
        "status",
        "reason",
        "baseline_file_count",
        "proposed_file_count",
        "graph_proposed_drops",
        "rerank_proposed_drops",
        "effective_proposed_drops",
        "rerank_status",
        "rerank_filter_status",
        "selection_unchanged",
        "elapsed_ms",
        "event_type",
        "context_status",
        "context_reason",
        "snapshot_gate_reason",
        "context_files_before",
        "context_files_after",
        "observation_point",
        "bundle_sha256_before",
        "bundle_sha256_after",
        "context_unchanged",
        "bundle_source_matched",
        "bundle_source_changed_paths",
        "bundle_source_sha256_before",
        "bundle_source_sha256_after",
        "case_id",
        "canary_sha256",
        "baseline_files",
        "candidate_files",
        "removed_files",
        "baseline_chars",
        "candidate_chars",
        "baseline_bundle_sha256",
        "selection_sha256",
        "source_sha256",
        "snapshot_matched",
        "preview_reason",
        "trial_reason",
        "audit_recorded",
        "production_ready",
        "active_execution_enabled",
        "consumed_bundle_sha256",
        "mandatory_context_complete",
        "retained_symbol_evidence",
        "metadata_unchanged",
        "scope_unchanged",
    )

    EVIDENCE_FIELDS = (
        "selection_sha256",
        "source_sha256_before",
        "source_sha256_after",
        "source_matched",
        "selection_matched",
        "changed_paths",
    )

    def __init__(
        self,
        *,
        path: str | Path,
    ) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def append(self, report: dict) -> None:
        event = {
            "event_id": uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # 普通审计字段
        for key in self.SIMPLE_FIELDS:
            if key in report:
                event[key] = report[key]

        # 证据版本绑定字段
        binding = report.get("evidence_binding")

        if isinstance(binding, dict):
            event["evidence_binding"] = {
                key: binding[key] for key in self.EVIDENCE_FIELDS if key in binding
            }

        # 不序列化 report 的其他未知字段
        line = json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
