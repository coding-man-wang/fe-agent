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
    TypeScript Reference Extractor。

    v1.4.4A：

        只提取 call_expression。

    其中：

        identifier()
            → 可解析

        member.expression()
            → 记录 occurrence
              但交给 4B。
    """

    adapter_id = "tree-sitter-typescript-references"

    _SUPPORTED_SUFFIXES = {
        ".ts",
        ".tsx",
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

        call_nodes: list[Node] = []

        self._collect_nodes(
            node=tree.root_node,
            node_type="call_expression",
            result=call_nodes,
        )

        references: list[SymbolReferenceOccurrence] = []

        for node in call_nodes:

            occurrence = self._extract_call(
                node=node,
                source=source,
                source_path=(document.relative_path),
            )

            if occurrence is not None:
                references.append(occurrence)

        return SymbolReferenceExtractionResult(
            adapter_id=self.adapter_id,
            relative_path=(document.relative_path),
            references=references,
            issues=[],
        )

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
            target_form = "identifier"

            name: str | None = expression_text

        elif function_node.type == "member_expression":
            target_form = "member"
            name = None

        else:
            target_form = "unknown"
            name = None

        return SymbolReferenceOccurrence(
            source_path=source_path,
            reference_kind="call",
            target_form=target_form,
            name=name,
            expression_text=(expression_text),
            line_number=(node.start_point.row + 1),
            column_number=(node.start_point.column + 1),
        )

    def _collect_nodes(
        self,
        *,
        node: Node,
        node_type: str,
        result: list[Node],
    ) -> None:

        if node.type == node_type:
            result.append(node)

        for child in node.named_children:

            self._collect_nodes(
                node=child,
                node_type=node_type,
                result=result,
            )

    @staticmethod
    def _node_text(
        *,
        node: Node,
        source: bytes,
    ) -> str:

        return source[node.start_byte : node.end_byte].decode("utf-8").strip()
