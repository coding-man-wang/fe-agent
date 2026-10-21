from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import Literal


ResolutionKind = Literal[
    "internal",
    "external",
    "unresolved_internal",
]


@dataclass(frozen=True)
class ImportResolution:
    kind: ResolutionKind
    target_path: str | None = None
    reason: str | None = None


class ModulePathResolver:
    """
    把 import specifier 解析成 Repository 内部文件路径。

    v1.2.1 支持：
    - 相对路径 ./ ../
    - @/ -> src/ alias
    - extension-less import
    - directory index import

    tsconfig paths 自动解析放到后续增强。
    """

    RESOLUTION_SUFFIXES = (
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".vue",
        ".json",
    )

    def __init__(
        self,
        *,
        file_paths: set[str],
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.file_paths = {
            self._normalize_path(path)
            for path in file_paths
        }

        self.aliases = aliases or {
            "@/": "src/",
        }

    def resolve(
        self,
        *,
        source_path: str,
        specifier: str,
    ) -> ImportResolution:
        normalized_specifier = self._strip_query(specifier.strip())

        base_path = self._internal_base_path(
            source_path=source_path,
            specifier=normalized_specifier,
        )

        if base_path is None:
            return ImportResolution(
                kind="external",
                reason="bare_module_specifier",
            )

        for candidate in self._candidates(base_path):
            if candidate in self.file_paths:
                return ImportResolution(
                    kind="internal",
                    target_path=candidate,
                )

        return ImportResolution(
            kind="unresolved_internal",
            reason=(
                "internal_module_path_not_found: "
                f"{normalized_specifier}"
            ),
        )

    def _internal_base_path(
        self,
        *,
        source_path: str,
        specifier: str,
    ) -> str | None:
        if specifier.startswith("."):
            source_directory = posixpath.dirname(
                self._normalize_path(source_path)
            )

            return self._normalize_path(
                posixpath.join(
                    source_directory,
                    specifier,
                )
            )

        for prefix, target_prefix in self.aliases.items():
            if specifier.startswith(prefix):
                remainder = specifier[len(prefix) :]

                return self._normalize_path(
                    posixpath.join(
                        target_prefix,
                        remainder,
                    )
                )

        return None

    def _candidates(
        self,
        base_path: str,
    ) -> list[str]:
        candidates: list[str] = [base_path]

        if not self._has_known_suffix(base_path):
            for suffix in self.RESOLUTION_SUFFIXES:
                candidates.append(base_path + suffix)

            for suffix in self.RESOLUTION_SUFFIXES:
                candidates.append(
                    posixpath.join(
                        base_path,
                        "index" + suffix,
                    )
                )

        return [
            self._normalize_path(item)
            for item in candidates
        ]

    def _has_known_suffix(
        self,
        path: str,
    ) -> bool:
        return any(
            path.endswith(suffix)
            for suffix in self.RESOLUTION_SUFFIXES
        )

    @staticmethod
    def _strip_query(specifier: str) -> str:
        return specifier.split("?", 1)[0].split("#", 1)[0]

    @staticmethod
    def _normalize_path(path: str) -> str:
        normalized = posixpath.normpath(
            path.replace("\\", "/")
        )

        if normalized == ".":
            return ""

        while normalized.startswith("./"):
            normalized = normalized[2:]

        return normalized
