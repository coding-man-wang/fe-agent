from __future__ import annotations

from collections.abc import MutableSet
from pathlib import PurePosixPath

import tree_sitter_typescript as ts_typescript
from tree_sitter import (
    Language,
    Node,
    Parser,
)

from agent_core.code_intelligence.model import (
    CodeIntelligenceIssue,
    CodeSymbol,
    SourceDocument,
    SymbolExtractionResult,
    SymbolKind,
    TechnologyProfile,
)

# ==================================
# 语法
#
# TypeScript 与 TSX 是两套不同 语法。
#
# .ts
#     → language_typescript()
#
# .tsx
#     → language_tsx()
#
# 不允许拿 TS 语法 强行解析 TSX。
# ==================================


_TYPESCRIPT_LANGUAGE = Language(ts_typescript.language_typescript())

_TSX_LANGUAGE = Language(ts_typescript.language_tsx())


class TypeScriptSymbolExtractorAdapter:
    """
    TypeScript / TSX Symbol Extraction Adapter。

    它负责：

        TypeScript Syntax
            ↓
        Common CodeSymbol

    上层 SymbolIndex /
    Retrieval /
    Impact Analysis

    都不需要知道 Tree-sitter。
    """

    adapter_id = "tree-sitter-typescript"

    priority = 10

    _SUPPORTED_SUFFIXES = {
        ".ts",
        ".tsx",
    }

    _FUNCTION_VALUE_TYPES = {
        "arrow_function",
        "function_expression",
    }

    def supports(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> bool:

        suffix = PurePosixPath(document.relative_path).suffix.lower()

        if suffix not in self._SUPPORTED_SUFFIXES:
            return False

        # Repository Discovery 已经确认 TS。
        if profile.has("typescript"):
            return True

        # 允许调用方显式指定 Document Language。
        document_language = (document.language or "").strip().lower()

        return document_language in {
            "typescript",
            "tsx",
        }

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
                "TypeScriptSymbolExtractorAdapter "
                "不支持该 SourceDocument: "
                f"{document.relative_path}"
            )

        parser = self._create_parser(document)

        source = document.content.encode("utf-8")

        tree = parser.parse(source)

        issues = self._collect_parse_issues(tree.root_node)

        symbols: list[CodeSymbol] = []

        allocated_ids: set[str] = set()

        self._walk(
            node=tree.root_node,
            source=source,
            relative_path=(document.relative_path),
            symbols=symbols,
            allocated_ids=allocated_ids,
            exported=False,
            parent_symbol_id=None,
            parent_qualified_name=None,
        )

        return SymbolExtractionResult(
            adapter_id=self.adapter_id,
            relative_path=(document.relative_path),
            symbols=symbols,
            issues=issues,
        )

    # ==================================
    # Parser Selection
    # ==================================
    def create_typescript_parser(
        document: SourceDocument,
    ) -> Parser:
        """
        TypeScript Adapter Layer 内部共享的
        AST Parser Factory。

        Symbol Extraction /
        Import Extraction /
        后续 Reference Extraction

        都复用这一入口。
        """

        suffix = PurePosixPath(document.relative_path).suffix.lower()

        if suffix == ".tsx":
            return Parser(_TSX_LANGUAGE)

        return Parser(_TYPESCRIPT_LANGUAGE)

    def _create_parser(
        self,
        document: SourceDocument,
    ) -> Parser:

        return create_typescript_parser(document)

    # ==================================
    # AST Traversal
    # ==================================

    def _walk(
        self,
        *,
        node: Node,
        source: bytes,
        relative_path: str,
        symbols: list[CodeSymbol],
        allocated_ids: MutableSet[str],
        exported: bool,
        parent_symbol_id: str | None,
        parent_qualified_name: str | None,
    ) -> None:

        # export_statement 自己不是 Symbol，
        # 但它下面的 declaration 是 exported。
        current_exported = exported or node.type == "export_statement"

        descriptor = self._describe_symbol(
            node=node,
            source=source,
        )

        current_parent_symbol_id = parent_symbol_id

        current_qualified_name = parent_qualified_name

        if descriptor is not None:

            (
                symbol_name,
                symbol_kind,
            ) = descriptor

            qualified_name = (
                symbol_name
                if not parent_qualified_name
                else (parent_qualified_name + "." + symbol_name)
            )

            symbol_id = self._allocate_symbol_id(
                relative_path=(relative_path),
                kind=symbol_kind,
                qualified_name=(qualified_name),
                allocated_ids=(allocated_ids),
            )

            symbol = CodeSymbol(
                symbol_id=symbol_id,
                name=symbol_name,
                kind=symbol_kind,
                relative_path=(relative_path),
                start_line=(node.start_point.row + 1),
                end_line=(node.end_point.row + 1),
                exported=(current_exported),
                parent_symbol_id=(parent_symbol_id),
            )

            symbols.append(symbol)

            current_parent_symbol_id = symbol.symbol_id

            current_qualified_name = qualified_name

        child_exported = current_exported if descriptor is None else False

        for child in node.named_children:
            self._walk(
                node=child,
                source=source,
                relative_path=relative_path,
                symbols=symbols,
                allocated_ids=allocated_ids,
                exported=child_exported,
                parent_symbol_id=(current_parent_symbol_id),
                parent_qualified_name=(current_qualified_name),
            )

    # ==================================
    # AST Node → Common Symbol
    # ==================================

    def _describe_symbol(
        self,
        *,
        node: Node,
        source: bytes,
    ) -> (
        tuple[
            str,
            SymbolKind,
        ]
        | None
    ):

        if node.type == "function_declaration":
            return self._named_symbol(
                node=node,
                source=source,
                kind="function",
            )

        if node.type == "class_declaration":
            return self._named_symbol(
                node=node,
                source=source,
                kind="class",
            )

        if node.type == "interface_declaration":
            return self._named_symbol(
                node=node,
                source=source,
                kind="interface",
            )

        if node.type == "type_alias_declaration":
            return self._named_symbol(
                node=node,
                source=source,
                kind="type",
            )

        if node.type == "method_definition":
            return self._named_symbol(
                node=node,
                source=source,
                kind="method",
            )

        if node.type == "variable_declarator":
            return self._describe_variable(
                node=node,
                source=source,
            )

        return None

    def _named_symbol(
        self,
        *,
        node: Node,
        source: bytes,
        kind: SymbolKind,
    ) -> (
        tuple[
            str,
            SymbolKind,
        ]
        | None
    ):

        name_node = node.child_by_field_name("name")

        if name_node is None:
            return None

        name = self._node_text(
            node=name_node,
            source=source,
        )

        if not name:
            return None

        return (
            name,
            kind,
        )

    def _describe_variable(
        self,
        *,
        node: Node,
        source: bytes,
    ) -> (
        tuple[
            str,
            SymbolKind,
        ]
        | None
    ):

        name_node = node.child_by_field_name("name")

        if name_node is None:
            return None

        # 第一版只索引简单 identifier：
        #
        # const foo = ...
        #
        # 暂不处理：
        #
        # const { foo } = ...
        # const [a, b] = ...
        #
        # destructuring symbol extraction
        # 进入后续 Backlog。
        if name_node.type != "identifier":
            return None

        name = self._node_text(
            node=name_node,
            source=source,
        )

        if not name:
            return None

        value_node = node.child_by_field_name("value")

        if value_node is not None and value_node.type in self._FUNCTION_VALUE_TYPES:
            kind: SymbolKind = "function"
        else:
            kind = "variable"

        return (
            name,
            kind,
        )

    # ==================================
    # Stable Symbol Identity
    # ==================================

    def _allocate_symbol_id(
        self,
        *,
        relative_path: str,
        kind: SymbolKind,
        qualified_name: str,
        allocated_ids: MutableSet[str],
    ) -> str:
        """
        默认 Symbol ID：

            path::kind::qualified-name

        示例：

            src/api/orders.ts
            ::function
            ::listOrders

        Class Method：

            src/services/orders.ts
            ::method
            ::OrderService.listOrders

        如果 TypeScript overload 等情况导致
        同 scope 出现同名 Symbol：

            #2
            #3

        只在 collision 时增加后缀。

        这样普通 Symbol 不会因为行号移动
        就改变 identity。
        """

        base_id = f"{relative_path}" f"::{kind}" f"::{qualified_name}"

        if base_id not in allocated_ids:
            allocated_ids.add(base_id)
            return base_id

        sequence = 2

        while True:

            candidate = f"{base_id}" f"#{sequence}"

            if candidate not in allocated_ids:
                allocated_ids.add(candidate)
                return candidate

            sequence += 1

    # ==================================
    # Syntax Error Diagnostics
    # ==================================

    def _collect_parse_issues(
        self,
        root: Node,
    ) -> list[CodeIntelligenceIssue]:

        if not root.has_error:
            return []

        issues: list[CodeIntelligenceIssue] = [
            CodeIntelligenceIssue(
                severity="warning",
                message=(
                    "Tree-sitter 检测到 "
                    "TypeScript Syntax Error；"
                    "已继续提取可恢复 Symbol"
                ),
            )
        ]

        self._collect_error_nodes(
            node=root,
            issues=issues,
            max_issues=10,
        )

        return issues

    def _collect_error_nodes(
        self,
        *,
        node: Node,
        issues: list[CodeIntelligenceIssue],
        max_issues: int,
    ) -> None:

        if len(issues) >= max_issues:
            return

        is_error = node.type == "ERROR"

        is_missing = bool(
            getattr(
                node,
                "is_missing",
                False,
            )
        )

        if is_error or is_missing:

            issues.append(
                CodeIntelligenceIssue(
                    severity="warning",
                    message=("Syntax recovery node: " f"{node.type}"),
                    line_number=(node.start_point.row + 1),
                )
            )

        for child in node.named_children:

            if len(issues) >= max_issues:
                break

            self._collect_error_nodes(
                node=child,
                issues=issues,
                max_issues=max_issues,
            )

    # ==================================
    # Source Text
    # ==================================

    @staticmethod
    def _node_text(
        *,
        node: Node,
        source: bytes,
    ) -> str:

        return source[node.start_byte : node.end_byte].decode("utf-8").strip()
