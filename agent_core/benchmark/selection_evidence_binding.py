from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from agent_core.codebase.model import (
    HybridRelevantFileSelection,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return sha256_bytes(encoded)


@dataclass(frozen=True)
class SelectionEvidenceFingerprint:
    selection_sha256: str
    source_sha256: str
    file_sha256: dict[str, str]


def capture_selection_evidence(
    *,
    repository_root: str | Path,
    selection: HybridRelevantFileSelection,
) -> SelectionEvidenceFingerprint:
    root = Path(repository_root).resolve(strict=True)

    if not root.is_dir():
        raise ValueError("Repository Root 不是目录")

    file_hashes: dict[str, str] = {}

    for item in selection.selected_files:
        relative_path = item.relative_path.replace("\\", "/")
        pure = PurePosixPath(relative_path)

        if (
            not pure.parts
            or pure.is_absolute()
            or PureWindowsPath(relative_path).drive
            or ".." in pure.parts
        ):
            raise ValueError(f"非法 Repository Path: {relative_path}")

        if relative_path in file_hashes:
            raise ValueError(f"重复文件: {relative_path}")

        source_path = (root / Path(*pure.parts)).resolve(strict=True)

        if not source_path.is_relative_to(root) or not source_path.is_file():
            raise ValueError("源码不属于当前 Repository")

        # 对真实文件字节计算哈希。
        # 不依赖可能过期的 Semantic Chunk。
        content = source_path.read_bytes()

        file_hashes[relative_path] = sha256_bytes(content)

    selection_hash = json_digest(selection.model_dump(mode="json"))

    source_hash = json_digest(sorted(file_hashes.items()))

    return SelectionEvidenceFingerprint(
        selection_sha256=selection_hash,
        source_sha256=source_hash,
        file_sha256=file_hashes,
    )
