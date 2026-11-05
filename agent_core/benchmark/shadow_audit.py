from __future__ import annotations

import hashlib
import json
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
    P1 策略审计记录。

    仅持久化白名单字段，避免意外写入
    Prompt、源码内容或模型凭据。
    """

    ALLOWED_FIELDS = frozenset(
        {
            "policy_version",
            "mode",
            "task_id",
            "repository_revision",
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
            "source_before_sha256",
            "source_after_sha256",
            "source_consistent",
            "selection_unchanged",
        }
    )

    def __init__(self, *, path: str | Path):
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

        for key in self.ALLOWED_FIELDS:
            if key in report:
                event[key] = report[key]

        line = json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(line + "\n")
