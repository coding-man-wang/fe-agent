from __future__ import annotations

import re
from pathlib import PurePosixPath

from agent_core.codebase.graph.model import ImportReference


class TypeScriptVueImportExtractor:
    """
    v1.2.1 的轻量 Import Extractor。

    当前只负责结构化提取：
    - import ... from 'x'
    - import type ... from 'x'
    - import 'x'
    - export ... from 'x'
    - import('x')

    Vue 文件只扫描 <script> / <script setup> 内容。

    这不是完整 TypeScript Parser。
    后续需要 symbol-level graph 时再引入 AST/tree-sitter。
    """

    SUPPORTED_SUFFIXES = {
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".vue",
    }

    _SCRIPT_BLOCK_PATTERN = re.compile(
        r"<script\b[^>]*>(?P<body>.*?)</script\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    _TYPE_IMPORT_PATTERN = re.compile(
        r"^[ \t]*import[ \t]+type\b.*?[ \t]+from[ \t]*"
        r"[\"'](?P<specifier>[^\"']+)[\"']",
        flags=re.MULTILINE | re.DOTALL,
    )

    _VALUE_IMPORT_PATTERN = re.compile(
        r"^[ \t]*import[ \t]+(?!type\b).*?[ \t]+from[ \t]*"
        r"[\"'](?P<specifier>[^\"']+)[\"']",
        flags=re.MULTILINE | re.DOTALL,
    )

    _SIDE_EFFECT_IMPORT_PATTERN = re.compile(
        r"^[ \t]*import[ \t]*[\"'](?P<specifier>[^\"']+)[\"']",
        flags=re.MULTILINE,
    )

    _REEXPORT_PATTERN = re.compile(
        r"^[ \t]*export[ \t]+(?:type[ \t]+)?(?:\*|\{).*?"
        r"[ \t]+from[ \t]*[\"'](?P<specifier>[^\"']+)[\"']",
        flags=re.MULTILINE | re.DOTALL,
    )

    _DYNAMIC_IMPORT_PATTERN = re.compile(
        r"\bimport\([ \t\r\n]*[\"'](?P<specifier>[^\"']+)[\"']"
        r"[ \t\r\n]*\)",
        flags=re.MULTILINE,
    )

    def extract(
        self,
        *,
        relative_path: str,
        content: str,
    ) -> list[ImportReference]:
        suffix = PurePosixPath(relative_path).suffix.lower()

        if suffix not in self.SUPPORTED_SUFFIXES:
            return []

        regions = self._regions(
            suffix=suffix,
            content=content,
        )

        collected: list[ImportReference] = []

        for region_content, absolute_offset in regions:
            collected.extend(
                self._extract_region(
                    source_path=relative_path,
                    full_content=content,
                    region_content=region_content,
                    absolute_offset=absolute_offset,
                )
            )

        unique: dict[tuple[str, str, int], ImportReference] = {}

        for item in collected:
            key = (
                item.specifier,
                item.import_kind,
                item.line_number,
            )
            unique[key] = item

        return sorted(
            unique.values(),
            key=lambda item: (
                item.line_number,
                item.specifier,
                item.import_kind,
            ),
        )

    def _regions(
        self,
        *,
        suffix: str,
        content: str,
    ) -> list[tuple[str, int]]:
        if suffix != ".vue":
            return [(content, 0)]

        regions: list[tuple[str, int]] = []

        for match in self._SCRIPT_BLOCK_PATTERN.finditer(content):
            regions.append(
                (
                    match.group("body"),
                    match.start("body"),
                )
            )

        return regions

    def _extract_region(
        self,
        *,
        source_path: str,
        full_content: str,
        region_content: str,
        absolute_offset: int,
    ) -> list[ImportReference]:
        patterns = [
            ("type", self._TYPE_IMPORT_PATTERN),
            ("value", self._VALUE_IMPORT_PATTERN),
            ("side_effect", self._SIDE_EFFECT_IMPORT_PATTERN),
            ("reexport", self._REEXPORT_PATTERN),
            ("dynamic", self._DYNAMIC_IMPORT_PATTERN),
        ]

        results: list[ImportReference] = []

        for import_kind, pattern in patterns:
            for match in pattern.finditer(region_content):
                absolute_start = absolute_offset + match.start()
                line_number = full_content.count(
                    "\n",
                    0,
                    absolute_start,
                ) + 1

                results.append(
                    ImportReference(
                        source_path=source_path,
                        specifier=match.group("specifier"),
                        import_kind=import_kind,
                        line_number=line_number,
                    )
                )

        return results
