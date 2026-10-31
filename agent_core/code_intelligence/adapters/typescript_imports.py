from __future__ import annotations

import re
from pathlib import PurePosixPath

from tree_sitter import Node

from agent_core.code_intelligence.adapters.typescript import (
    create_typescript_parser,
)
from agent_core.code_intelligence.model import (
    ImportBinding,
    ImportBindingExtractionResult,
    SourceDocument,
    TechnologyProfile,
)


class TypeScriptImportBindingExtractorAdapter:
    """
    TypeScript AST Import Binding Extractor。

    注意：

    旧 TypeScriptVueImportExtractor
    负责 file-level graph。

    本类负责：
        symbol-level import binding。

    两者暂时并存，
    避免破坏 v1.2 graph。
    """

    adapter_id = "tree-sitter-typescript-imports"

    _SUPPORTED_SUFFIXES = {
        ".ts",
        ".tsx",
    }

    _FROM_PATTERN = re.compile(
        r"""
        \bfrom\s*
        (?P<quote>["'])
        (?P<specifier>.*?)
        (?P=quote)
        """,
        re.VERBOSE | re.DOTALL,
    )

    _SIDE_EFFECT_PATTERN = re.compile(
        r"""
        ^\s*
        import\s*
        (?P<quote>["'])
        (?P<specifier>.*?)
        (?P=quote)
        """,
        re.VERBOSE | re.DOTALL,
    )

    _NAMESPACE_PATTERN = re.compile(
        r"""
        ^\s*
        \*\s+as\s+
        (?P<local>[A-Za-z_$][\w$]*)
        \s*$
        """,
        re.VERBOSE,
    )

    def supports(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> bool:

        suffix = PurePosixPath(document.relative_path).suffix.lower()

        if suffix not in self._SUPPORTED_SUFFIXES:
            return False

        if profile.has("typescript"):
            return True

        return (document.language or "").strip().lower() in {
            "typescript",
            "tsx",
        }

    def extract_import_bindings(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> ImportBindingExtractionResult:

        if not self.supports(
            profile=profile,
            document=document,
        ):
            raise ValueError(
                "TypeScript Import Extractor "
                "不支持该 SourceDocument: " + document.relative_path
            )

        parser = create_typescript_parser(document)

        source = document.content.encode("utf-8")

        tree = parser.parse(source)

        import_nodes: list[Node] = []

        self._collect_nodes(
            node=tree.root_node,
            node_type="import_statement",
            result=import_nodes,
        )

        bindings: list[ImportBinding] = []

        for node in import_nodes:
            bindings.extend(
                self._extract_statement(
                    node=node,
                    source=source,
                    source_path=(document.relative_path),
                )
            )

        return ImportBindingExtractionResult(
            adapter_id=self.adapter_id,
            relative_path=(document.relative_path),
            bindings=bindings,
            issues=[],
        )

    # ==================================
    # Import Statement
    # ==================================

    def _extract_statement(
        self,
        *,
        node: Node,
        source: bytes,
        source_path: str,
    ) -> list[ImportBinding]:

        statement = self._node_text(
            node=node,
            source=source,
        )

        line_number = node.start_point.row + 1

        from_match = self._FROM_PATTERN.search(statement)

        # ----------------------------------
        # Side-effect import
        #
        # import "./setup"
        # ----------------------------------

        if from_match is None:

            side_effect = self._SIDE_EFFECT_PATTERN.search(statement)

            if side_effect is None:
                return []

            return [
                ImportBinding(
                    source_path=source_path,
                    module_specifier=(side_effect.group("specifier")),
                    binding_kind=("side_effect"),
                    imported_name=None,
                    local_name=None,
                    type_only=False,
                    line_number=line_number,
                )
            ]

        module_specifier = from_match.group("specifier")

        statement_type_only = bool(
            re.match(
                r"^\s*import\s+type\b",
                statement,
            )
        )

        clause = self._find_first_descendant(
            node,
            "import_clause",
        )

        if clause is None:
            return []

        bindings: list[ImportBinding] = []

        # ----------------------------------
        # Default Import
        #
        # import OrdersApi from "./orders"
        # ----------------------------------

        for child in clause.named_children:

            if child.type == "identifier":

                local_name = self._node_text(
                    node=child,
                    source=source,
                )

                bindings.append(
                    ImportBinding(
                        source_path=source_path,
                        module_specifier=(module_specifier),
                        binding_kind="default",
                        imported_name="default",
                        local_name=local_name,
                        type_only=(statement_type_only),
                        line_number=(line_number),
                    )
                )

                # 一个 import_clause 最多一个
                # default binding。
                break

        # ----------------------------------
        # Namespace Import
        #
        # import * as OrdersApi
        # from "./orders"
        # ----------------------------------

        namespace = self._find_first_descendant(
            clause,
            "namespace_import",
        )

        if namespace is not None:

            text = self._node_text(
                node=namespace,
                source=source,
            )

            match = self._NAMESPACE_PATTERN.match(text)

            if match is not None:

                bindings.append(
                    ImportBinding(
                        source_path=source_path,
                        module_specifier=(module_specifier),
                        binding_kind=("namespace"),
                        imported_name="*",
                        local_name=(match.group("local")),
                        type_only=(statement_type_only),
                        line_number=(line_number),
                    )
                )

        # ----------------------------------
        # Named Import
        #
        # import {
        #   listOrders,
        #   listUsers as fetchUsers,
        #   type Order
        # } from "..."
        # ----------------------------------

        named_imports: list[Node] = []

        self._collect_nodes(
            node=clause,
            node_type="import_specifier",
            result=named_imports,
        )

        for specifier in named_imports:

            parsed = self._parse_named_import(
                node=specifier,
                source=source,
                statement_type_only=(statement_type_only),
            )

            if parsed is None:
                continue

            (
                imported_name,
                local_name,
                type_only,
            ) = parsed

            bindings.append(
                ImportBinding(
                    source_path=source_path,
                    module_specifier=(module_specifier),
                    binding_kind="named",
                    imported_name=(imported_name),
                    local_name=local_name,
                    type_only=type_only,
                    line_number=(specifier.start_point.row + 1),
                )
            )

        return bindings

    def _parse_named_import(
        self,
        *,
        node: Node,
        source: bytes,
        statement_type_only: bool,
    ) -> (
        tuple[
            str,
            str,
            bool,
        ]
        | None
    ):

        text = self._node_text(
            node=node,
            source=source,
        ).strip()

        specifier_type_only = statement_type_only

        if text.startswith("type "):
            specifier_type_only = True
            text = text[len("type ") :].strip()

        parts = re.split(
            r"\s+as\s+",
            text,
            maxsplit=1,
        )

        imported_name = parts[0].strip()

        if not imported_name:
            return None

        local_name = parts[1].strip() if len(parts) == 2 else imported_name

        if not local_name:
            return None

        return (
            imported_name,
            local_name,
            specifier_type_only,
        )

    # ==================================
    # AST Utilities
    # ==================================

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

    def _find_first_descendant(
        self,
        node: Node,
        node_type: str,
    ) -> Node | None:

        if node.type == node_type:
            return node

        for child in node.named_children:

            result = self._find_first_descendant(
                child,
                node_type,
            )

            if result is not None:
                return result

        return None

    @staticmethod
    def _node_text(
        *,
        node: Node,
        source: bytes,
    ) -> str:

        return source[node.start_byte : node.end_byte].decode("utf-8")
