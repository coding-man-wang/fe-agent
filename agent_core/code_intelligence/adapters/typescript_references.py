from __future__ import annotations

from pathlib import PurePosixPath

from tree_sitter import Node

from agent_core.code_intelligence.adapters.typescript import (
    create_typescript_parser,
)
from agent_core.code_intelligence.model import (
    SourceDocument,
    SymbolReferenceExtractionResult,
    SymbolReferenceOccurrence,
    TechnologyProfile,
)


class TypeScriptSymbolReferenceExtractor:
    """
    TypeScript / TSX Reference Extractor。

    v1.4.4B 支持：

        identifier call
        member call
        identifier read
        member read
        type reference

    这里只负责：

        AST
        ↓
        Reference Occurrence

    不负责 Symbol Resolution。
    """

    adapter_id = "tree-sitter-typescript-references"

    _SUPPORTED_SUFFIXES = {
        ".ts",
        ".tsx",
    }

    _DECLARATION_TYPES = {
        "function_declaration",
        "function_signature",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "method_definition",
        "variable_declarator",
    }

    _PARAMETER_TYPES = {
        "required_parameter",
        "optional_parameter",
        "rest_pattern",
    }

    _SKIP_SUBTREES = {
        # Import 本身不是 Reference Usage。
        "import_statement",
        # Export Specifier 本身也不是
        # Runtime/Type Reference Usage。
        "export_clause",
        "export_specifier",
        "namespace_export",
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

        return profile.has("typescript") or (
            document.language or ""
        ).strip().lower() in {
            "typescript",
            "tsx",
        }

    def extract_references(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> SymbolReferenceExtractionResult:

        if not self.supports(
            profile=profile,
            document=document,
        ):
            raise ValueError(
                "TypeScript Reference Extractor "
                "不支持该文件: " + document.relative_path
            )

        parser = create_typescript_parser(document)

        source = document.content.encode("utf-8")

        tree = parser.parse(source)

        references: list[SymbolReferenceOccurrence] = []

        self._walk(
            node=tree.root_node,
            parent=None,
            source=source,
            source_path=(document.relative_path),
            result=references,
        )

        return SymbolReferenceExtractionResult(
            adapter_id=self.adapter_id,
            relative_path=(document.relative_path),
            references=references,
            issues=[],
        )

    # ==================================
    # AST Traversal
    # ==================================

    def _walk(
        self,
        *,
        node: Node,
        parent: Node | None,
        source: bytes,
        source_path: str,
        result: list[SymbolReferenceOccurrence],
    ) -> None:

        if node.type in self._SKIP_SUBTREES:
            return

        # ==================================
        # Call Expression
        #
        # 先把 callee 整体抽成一个
        # Reference Occurrence。
        #
        # 然后只继续扫描 arguments，
        # 防止：
        #
        # listOrders()
        #
        # 同时产生：
        #   call(listOrders)
        #   read(listOrders)
        # ==================================

        if node.type == "call_expression":

            occurrence = self._extract_call(
                node=node,
                source=source,
                source_path=(source_path),
            )

            if occurrence is not None:
                result.append(occurrence)

            function_node = node.child_by_field_name("function")

            for child in node.named_children:

                if function_node is not None and self._same_node(
                    child,
                    function_node,
                ):
                    continue

                self._walk(
                    node=child,
                    parent=node,
                    source=source,
                    source_path=(source_path),
                    result=result,
                )

            return

        # ==================================
        # Standalone Member Read
        #
        # const handler = api.listOrders
        #
        # 但：
        #
        # api.listOrders()
        #
        # 已由 call_expression 处理。
        # ==================================

        if node.type == "member_expression":

            if not (parent is not None and parent.type == "member_expression"):

                occurrence = self._extract_member(
                    node=node,
                    source=source,
                    source_path=(source_path),
                    reference_kind="read",
                )

                if occurrence is not None:
                    result.append(occurrence)

            # 不再递归 property，
            # 防止 listOrders 被再次当成
            # identifier read。
            #
            # 对复杂 object：
            #
            # getApi().listOrders
            #
            # 仍然扫描 getApi()。
            object_node = node.child_by_field_name("object")

            if object_node is not None and object_node.type not in {
                "identifier",
                "this",
            }:
                self._walk(
                    node=object_node,
                    parent=node,
                    source=source,
                    source_path=(source_path),
                    result=result,
                )

            return

        # ==================================
        # Type Reference
        # ==================================

        if node.type == "type_identifier":

            if not self._is_definition_name(
                node=node,
                parent=parent,
            ):

                text = self._node_text(
                    node=node,
                    source=source,
                )

                if text:

                    result.append(
                        SymbolReferenceOccurrence(
                            source_path=(source_path),
                            reference_kind=("type_reference"),
                            target_form=("identifier"),
                            name=text,
                            expression_text=text,
                            line_number=(node.start_point.row + 1),
                            column_number=(node.start_point.column + 1),
                        )
                    )

            return

        # ==================================
        # Ordinary Identifier Read
        # ==================================

        if node.type == "identifier" and self._is_read_identifier(
            node=node,
            parent=parent,
        ):

            text = self._node_text(
                node=node,
                source=source,
            )

            if text:

                result.append(
                    SymbolReferenceOccurrence(
                        source_path=(source_path),
                        reference_kind="read",
                        target_form="identifier",
                        name=text,
                        expression_text=text,
                        line_number=(node.start_point.row + 1),
                        column_number=(node.start_point.column + 1),
                    )
                )

        for child in node.named_children:

            self._walk(
                node=child,
                parent=node,
                source=source,
                source_path=(source_path),
                result=result,
            )

    # ==================================
    # Call
    # ==================================

    def _extract_call(
        self,
        *,
        node: Node,
        source: bytes,
        source_path: str,
    ) -> SymbolReferenceOccurrence | None:

        function_node = node.child_by_field_name("function")

        if function_node is None:
            return None

        expression_text = self._node_text(
            node=function_node,
            source=source,
        )

        if not expression_text:
            return None

        if function_node.type == "identifier":

            return SymbolReferenceOccurrence(
                source_path=source_path,
                reference_kind="call",
                target_form="identifier",
                name=expression_text,
                expression_text=(expression_text),
                line_number=(node.start_point.row + 1),
                column_number=(node.start_point.column + 1),
            )

        if function_node.type == "member_expression":

            return self._extract_member(
                node=function_node,
                source=source,
                source_path=source_path,
                reference_kind="call",
                line_number=(node.start_point.row + 1),
                column_number=(node.start_point.column + 1),
            )

        return SymbolReferenceOccurrence(
            source_path=source_path,
            reference_kind="call",
            target_form="unknown",
            name=None,
            expression_text=(expression_text),
            line_number=(node.start_point.row + 1),
            column_number=(node.start_point.column + 1),
        )

    # ==================================
    # Member Expression
    # ==================================

    def _extract_member(
        self,
        *,
        node: Node,
        source: bytes,
        source_path: str,
        reference_kind: str,
        line_number: int | None = None,
        column_number: int | None = None,
    ) -> SymbolReferenceOccurrence | None:

        object_node = node.child_by_field_name("object")

        property_node = node.child_by_field_name("property")

        if object_node is None or property_node is None:
            return None

        expression_text = self._node_text(
            node=node,
            source=source,
        )

        member_name = self._node_text(
            node=property_node,
            source=source,
        )

        receiver_name: str | None

        if object_node.type in {
            "identifier",
            "this",
        }:
            receiver_name = self._node_text(
                node=object_node,
                source=source,
            )
        else:
            receiver_name = None

        return SymbolReferenceOccurrence(
            source_path=source_path,
            reference_kind=reference_kind,
            target_form="member",
            name=member_name or None,
            receiver_name=(receiver_name),
            member_name=(member_name or None),
            expression_text=(expression_text),
            line_number=(line_number or (node.start_point.row + 1)),
            column_number=(column_number or (node.start_point.column + 1)),
        )

    # ==================================
    # Identifier Classification
    # ==================================

    def _is_read_identifier(
        self,
        *,
        node: Node,
        parent: Node | None,
    ) -> bool:

        if parent is None:
            return False

        if self._is_definition_name(
            node=node,
            parent=parent,
        ):
            return False

        if parent.type in self._PARAMETER_TYPES:
            return False

        if parent.type in {
            "import_specifier",
            "namespace_import",
            "export_specifier",
            "label_name",
        }:
            return False

        return True

    def _is_definition_name(
        self,
        *,
        node: Node,
        parent: Node | None,
    ) -> bool:

        if parent is None:
            return False

        if parent.type not in self._DECLARATION_TYPES:
            return False

        name_node = parent.child_by_field_name("name")

        if name_node is None:
            return False

        return self._same_node(
            node,
            name_node,
        )

    # ==================================
    # Node Utilities
    # ==================================

    @staticmethod
    def _same_node(
        left: Node,
        right: Node,
    ) -> bool:

        return (
            left.start_byte == right.start_byte
            and left.end_byte == right.end_byte
            and left.type == right.type
        )

    @staticmethod
    def _node_text(
        *,
        node: Node,
        source: bytes,
    ) -> str:

        return source[node.start_byte : node.end_byte].decode("utf-8").strip()
