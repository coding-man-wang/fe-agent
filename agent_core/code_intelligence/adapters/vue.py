from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from agent_core.code_intelligence.adapters.typescript import (
    TypeScriptSymbolExtractorAdapter,
)
from agent_core.code_intelligence.model import (
    CodeIntelligenceIssue,
    CodeSymbol,
    SourceDocument,
    SymbolExtractionResult,
    TechnologyProfile,
)

VueScriptKind = Literal[
    "script",
    "script_setup",
]


@dataclass(
    frozen=True,
    slots=True,
)
class _VueScriptBlock:
    """
    Vue SFC 内部的 Script Fragment。

    注意：

    它只是 Vue Adapter 内部模型，
    不应该泄漏到 Code Intelligence Core。
    """

    kind: VueScriptKind

    content: str

    language: str

    # Script content 相对于整个
    # .vue 文件的行号偏移。
    line_offset: int

    has_src: bool = False


class VueSfcSymbolExtractorAdapter:
    """
    Vue Single-File Component
    Symbol Extraction Adapter。

    主要职责：

        Vue SFC
            ↓
        Script Fragment
            ↓
        TypeScript Adapter
            ↓
        Common CodeSymbol

    Vue Adapter 不负责解析
    TypeScript AST。
    """

    adapter_id = "vue-sfc"

    # 比 generic TypeScript Adapter 高，
    # 因为 .vue 是更具体的 Container。
    priority = 100

    def __init__(
        self,
        *,
        typescript_adapter: TypeScriptSymbolExtractorAdapter | None = None,
    ) -> None:

        self._typescript_adapter = (
            typescript_adapter or TypeScriptSymbolExtractorAdapter()
        )

    # ==================================
    # Adapter Selection
    # ==================================

    def supports(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> bool:

        suffix = PurePosixPath(document.relative_path).suffix.lower()

        if suffix != ".vue":
            return False

        # Nuxt 本质上同样使用 Vue SFC。
        #
        # Framework-specific knowledge
        # 保留在 Adapter Boundary 内，
        # 不泄漏给 SymbolIndex / Retrieval。
        return profile.has("vue") or profile.has("nuxt")

    # ==================================
    # Public Extraction
    # ==================================

    def extract_symbols(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> SymbolExtractionResult:

        if not self.supports(
            profile=profile,
            document=document,
        ):
            raise ValueError(
                "VueSfcSymbolExtractorAdapter "
                "不支持该 SourceDocument: "
                f"{document.relative_path}"
            )

        issues: list[CodeIntelligenceIssue] = []

        component = self._build_component_symbol(document)

        symbols: list[CodeSymbol] = [component]

        blocks = self._extract_script_blocks(document.content)

        if not blocks:

            # Template-only component
            # 本身仍然是合法 Component。
            return SymbolExtractionResult(
                adapter_id=self.adapter_id,
                relative_path=(document.relative_path),
                symbols=symbols,
                issues=[],
            )

        for block_index, block in enumerate(
            blocks,
            start=1,
        ):

            if block.has_src:

                issues.append(
                    CodeIntelligenceIssue(
                        severity="warning",
                        message=(
                            "Vue external script src "
                            "暂未进入 v1.4.2 "
                            "Code Intelligence"
                        ),
                        line_number=(block.line_offset + 1),
                    )
                )

                continue

            if block.language not in {
                "ts",
                "tsx",
            }:

                issues.append(
                    CodeIntelligenceIssue(
                        severity="warning",
                        message=("Vue Script Language " "暂未支持: " + block.language),
                        line_number=(block.line_offset + 1),
                    )
                )

                continue

            result = self._extract_typescript_block(
                profile=profile,
                document=document,
                block=block,
                block_index=block_index,
                component=component,
            )

            symbols.extend(result.symbols)

            issues.extend(result.issues)

        return SymbolExtractionResult(
            adapter_id=self.adapter_id,
            relative_path=(document.relative_path),
            symbols=symbols,
            issues=issues,
        )

    # ==================================
    # Component Symbol
    # ==================================

    def _build_component_symbol(
        self,
        document: SourceDocument,
    ) -> CodeSymbol:

        component_name = PurePosixPath(document.relative_path).stem

        total_lines = max(
            1,
            (document.content.count("\n") + 1),
        )

        return CodeSymbol(
            symbol_id=(
                f"{document.relative_path}" f"::component" f"::{component_name}"
            ),
            name=component_name,
            kind="component",
            relative_path=(document.relative_path),
            start_line=1,
            end_line=total_lines,
            # 一个 .vue 文件对外暴露的核心
            # Module Symbol 就是 Component。
            exported=True,
            parent_symbol_id=None,
        )

    # ==================================
    # Embedded TypeScript
    # ==================================

    def _extract_typescript_block(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
        block: _VueScriptBlock,
        block_index: int,
        component: CodeSymbol,
    ) -> SymbolExtractionResult:

        virtual_suffix = ".tsx" if block.language == "tsx" else ".ts"

        virtual_path = (
            f"{document.relative_path}"
            f".__{block.kind}_"
            f"{block_index}"
            f"{virtual_suffix}"
        )

        virtual_document = SourceDocument(
            relative_path=(virtual_path),
            content=block.content,
            language=("tsx" if block.language == "tsx" else "typescript"),
        )

        typescript_result = self._typescript_adapter.extract_symbols(
            profile=profile,
            document=(virtual_document),
        )

        remapped_symbols = self._remap_symbols(
            original_path=(document.relative_path),
            block=block,
            block_index=block_index,
            component=component,
            source_symbols=(typescript_result.symbols),
        )

        remapped_issues = [
            issue.model_copy(
                deep=True,
                update={
                    "line_number": (
                        (issue.line_number + block.line_offset)
                        if (issue.line_number is not None)
                        else None
                    )
                },
            )
            for issue in typescript_result.issues
        ]

        return SymbolExtractionResult(
            adapter_id=self.adapter_id,
            relative_path=(document.relative_path),
            symbols=remapped_symbols,
            issues=remapped_issues,
        )

    # ==================================
    # Virtual Symbol → Vue Symbol
    # ==================================

    def _remap_symbols(
        self,
        *,
        original_path: str,
        block: _VueScriptBlock,
        block_index: int,
        component: CodeSymbol,
        source_symbols: list[CodeSymbol],
    ) -> list[CodeSymbol]:

        id_map: dict[
            str,
            str,
        ] = {}

        # ----------------------------------
        # Script Scope 只参与 Symbol Identity，
        # 不暴露成额外 Symbol。
        #
        # script_setup.listOrders
        #
        # 这样普通 script 和 script setup
        # 即使存在同名 binding，也不会撞 ID。
        # ----------------------------------

        scope_name = block.kind

        if block_index > 1:
            scope_name = f"{scope_name}" f"_{block_index}"

        for symbol in source_symbols:

            qualified_name = self._extract_qualified_name(symbol.symbol_id)

            new_id = (
                f"{original_path}"
                f"::{symbol.kind}"
                f"::{scope_name}"
                f".{qualified_name}"
            )

            id_map[symbol.symbol_id] = new_id

        result: list[CodeSymbol] = []

        for symbol in source_symbols:

            if symbol.parent_symbol_id is None:
                parent_symbol_id = component.symbol_id
            else:
                parent_symbol_id = id_map.get(
                    symbol.parent_symbol_id,
                    component.symbol_id,
                )

            result.append(
                symbol.model_copy(
                    deep=True,
                    update={
                        "symbol_id": (id_map[symbol.symbol_id]),
                        "relative_path": (original_path),
                        "start_line": (symbol.start_line + block.line_offset),
                        "end_line": (symbol.end_line + block.line_offset),
                        "parent_symbol_id": (parent_symbol_id),
                    },
                )
            )

        return result

    @staticmethod
    def _extract_qualified_name(
        symbol_id: str,
    ) -> str:

        parts = symbol_id.split(
            "::",
            2,
        )

        if len(parts) != 3:
            return symbol_id

        return parts[2]

    # ==================================
    # SFC Script Block Extraction
    # ==================================

    def _extract_script_blocks(
        self,
        source: str,
    ) -> list[_VueScriptBlock]:
        """
        只负责提取 Vue SFC Script Container。

        本阶段没有引入完整 Vue Parser。

        这里使用 deterministic scanner，
        而不是简单：

            <script[^>]*>

        Regex。

        原因：

        Vue 支持类似：

            generic="T extends Foo<Bar>"

        属性内容可能包含 >。

        单纯 [^>]* 会提前终止。
        """

        lower_source = source.lower()

        blocks: list[_VueScriptBlock] = []

        cursor = 0

        while True:

            start = lower_source.find(
                "<script",
                cursor,
            )

            if start < 0:
                break

            after_name = start + len("<script")

            # 防止：
            #
            # <scripted>
            #
            # 被误识别为 script。
            if after_name < len(source) and not (
                source[after_name].isspace() or source[after_name] == ">"
            ):
                cursor = after_name
                continue

            open_end = self._find_tag_end(
                source,
                after_name,
            )

            if open_end is None:
                break

            attrs = source[after_name:open_end]

            close_start = lower_source.find(
                "</script",
                open_end + 1,
            )

            if close_start < 0:
                # 正在编辑中的不完整 SFC。
                #
                # 不让整个 Repository
                # Code Intelligence 崩溃。
                break

            close_end = self._find_tag_end(
                source,
                (close_start + len("</script")),
            )

            if close_end is None:
                break

            content_start = open_end + 1

            content = source[content_start:close_start]

            kind: VueScriptKind = (
                "script_setup"
                if self._has_attribute(
                    attrs,
                    "setup",
                )
                else "script"
            )

            language = (
                self._attribute_value(
                    attrs,
                    "lang",
                )
                or "js"
            ).lower()

            has_src = self._has_attribute(
                attrs,
                "src",
            )

            line_offset = source[:content_start].count("\n")

            blocks.append(
                _VueScriptBlock(
                    kind=kind,
                    content=content,
                    language=language,
                    line_offset=line_offset,
                    has_src=has_src,
                )
            )

            cursor = close_end + 1

        return blocks

    # ==================================
    # Start Tag Scanner
    # ==================================

    @staticmethod
    def _find_tag_end(
        source: str,
        start: int,
    ) -> int | None:

        quote: str | None = None

        index = start

        while index < len(source):

            char = source[index]

            if quote is not None:

                if char == quote:
                    quote = None

                index += 1
                continue

            if char in {
                '"',
                "'",
            }:
                quote = char
                index += 1
                continue

            if char == ">":
                return index

            index += 1

        return None

    # ==================================
    # Attribute Parsing
    # ==================================

    @staticmethod
    def _has_attribute(
        attrs: str,
        name: str,
    ) -> bool:

        pattern = re.compile(
            rf"""
            (?:
                ^
                |
                \s
            )
            {re.escape(name)}
            (?:
                \s
                |
                =
                |
                $
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        return bool(pattern.search(attrs))

    @staticmethod
    def _attribute_value(
        attrs: str,
        name: str,
    ) -> str | None:

        pattern = re.compile(
            rf"""
            (?:
                ^
                |
                \s
            )
            {re.escape(name)}
            \s*
            =
            \s*
            (?:
                "([^"]*)"
                |
                '([^']*)'
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        match = pattern.search(attrs)

        if match is None:
            return None

        return match.group(1) if (match.group(1) is not None) else match.group(2)
