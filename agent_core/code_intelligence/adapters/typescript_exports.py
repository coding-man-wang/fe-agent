from __future__ import annotations

import re
from pathlib import PurePosixPath

from tree_sitter import Node

from agent_core.code_intelligence.adapters.typescript import (
    create_typescript_parser,
)
from agent_core.code_intelligence.model import (
    CodeIntelligenceIssue,
    ModuleExportBinding,
    ModuleExportExtractionResult,
    SourceDocument,
    TechnologyProfile,
)


class TypeScriptModuleExportExtractor:
    """
    TypeScript Module Export Extractor。

    Source
        ↓
    export_statement AST
        ↓
    ModuleExportBinding[]
    """

    adapter_id = "tree-sitter-typescript-exports"

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

    def supports(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> bool:

        suffix = PurePosixPath(document.relative_path).suffix.lower()

        return suffix in self._SUPPORTED_SUFFIXES and (
            profile.has("typescript")
            or (document.language or "").lower()
            in {
                "typescript",
                "tsx",
            }
        )

    def extract_module_exports(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> ModuleExportExtractionResult:

        if not self.supports(
            profile=profile,
            document=document,
        ):
            raise ValueError(
                "TypeScript Export Extractor " "不支持该文件: " + document.relative_path
            )

        parser = create_typescript_parser(document)

        source = document.content.encode("utf-8")

        tree = parser.parse(source)

        nodes: list[Node] = []

        self._collect_nodes(
            node=tree.root_node,
            node_type="export_statement",
            result=nodes,
        )

        exports: list[ModuleExportBinding] = []

        issues: list[CodeIntelligenceIssue] = []

        for node in nodes:

            bindings, node_issues = self._extract_statement(
                node=node,
                source=source,
                source_path=(document.relative_path),
            )

            exports.extend(bindings)

            issues.extend(node_issues)

        return ModuleExportExtractionResult(
            adapter_id=self.adapter_id,
            relative_path=(document.relative_path),
            exports=exports,
            issues=issues,
        )

    def _extract_statement(
        self,
        *,
        node: Node,
        source: bytes,
        source_path: str,
    ) -> tuple[
        list[ModuleExportBinding],
        list[CodeIntelligenceIssue],
    ]:

        text = self._node_text(
            node=node,
            source=source,
        ).strip()

        line_number = node.start_point.row + 1

        issues: list[CodeIntelligenceIssue] = []

        module_specifier = self._module_specifier(text)

        statement_type_only = text.startswith("export type ")

        # ==================================
        # export * as api from "./api"
        #
        # namespace re-export 暂不处理。
        # ==================================

        if re.match(
            r"^export\s+\*\s+as\s+",
            text,
        ):

            issues.append(
                CodeIntelligenceIssue(
                    severity="warning",
                    message=("namespace re-export " "暂未支持"),
                    line_number=line_number,
                )
            )

            return (
                [],
                issues,
            )

        # ==================================
        # export * from "./orders"
        # ==================================

        if re.match(
            r"^export\s+\*\s+from\b",
            text,
        ):

            if module_specifier:

                return (
                    [
                        ModuleExportBinding(
                            source_path=(source_path),
                            binding_kind=("star_reexport"),
                            module_specifier=(module_specifier),
                            line_number=(line_number),
                            type_only=(statement_type_only),
                        )
                    ],
                    issues,
                )

        # ==================================
        # export { foo }
        # export { foo as bar }
        # export { foo } from "./x"
        # ==================================

        specifiers: list[Node] = []

        self._collect_nodes(
            node=node,
            node_type="export_specifier",
            result=specifiers,
        )

        if specifiers:

            result: list[ModuleExportBinding] = []

            for specifier in specifiers:

                parsed = self._parse_specifier(
                    node=specifier,
                    source=source,
                    statement_type_only=(statement_type_only),
                )

                if parsed is None:
                    continue

                (
                    source_name,
                    exported_name,
                    type_only,
                ) = parsed

                if module_specifier:

                    result.append(
                        ModuleExportBinding(
                            source_path=(source_path),
                            binding_kind=("reexport"),
                            exported_name=(exported_name),
                            imported_name=(source_name),
                            module_specifier=(module_specifier),
                            type_only=type_only,
                            line_number=(specifier.start_point.row + 1),
                        )
                    )

                else:

                    result.append(
                        ModuleExportBinding(
                            source_path=(source_path),
                            binding_kind="local",
                            exported_name=(exported_name),
                            local_name=(source_name),
                            type_only=type_only,
                            line_number=(specifier.start_point.row + 1),
                        )
                    )

            return (
                result,
                issues,
            )

        # ==================================
        # export default ...
        # ==================================

        if text.startswith("export default"):

            local_name = self._find_declaration_name(
                node=node,
                source=source,
            )

            if local_name is None:

                issues.append(
                    CodeIntelligenceIssue(
                        severity="warning",
                        message=(
                            "anonymous default export " "暂时无法映射到 CodeSymbol"
                        ),
                        line_number=(line_number),
                    )
                )

            return (
                [
                    ModuleExportBinding(
                        source_path=(source_path),
                        binding_kind="local",
                        exported_name="default",
                        local_name=local_name,
                        line_number=line_number,
                    )
                ],
                issues,
            )

        # ==================================
        # export function foo
        # export class Foo
        # export interface Foo
        # export type Foo
        # export const foo = ...
        # ==================================

        names = self._find_exported_declaration_names(
            node=node,
            source=source,
        )

        return (
            [
                ModuleExportBinding(
                    source_path=source_path,
                    binding_kind="local",
                    exported_name=name,
                    local_name=name,
                    type_only=(statement_type_only),
                    line_number=line_number,
                )
                for name in names
            ],
            issues,
        )

    # ==================================
    # Specifier
    # ==================================

    def _parse_specifier(
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

        type_only = statement_type_only

        if text.startswith("type "):
            type_only = True

            text = text[len("type ") :].strip()

        parts = re.split(
            r"\s+as\s+",
            text,
            maxsplit=1,
        )

        source_name = parts[0].strip()

        if not source_name:
            return None

        exported_name = parts[1].strip() if len(parts) == 2 else source_name

        return (
            source_name,
            exported_name,
            type_only,
        )

    # ==================================
    # Export Declaration Names
    # ==================================

    def _find_declaration_name(
        self,
        *,
        node: Node,
        source: bytes,
    ) -> str | None:

        supported = {
            "function_declaration",
            "class_declaration",
        }

        for child in node.named_children:

            if child.type not in supported:
                continue

            name_node = child.child_by_field_name("name")

            if name_node is not None:

                return self._node_text(
                    node=name_node,
                    source=source,
                ).strip()

        return None

    def _find_exported_declaration_names(
        self,
        *,
        node: Node,
        source: bytes,
    ) -> list[str]:

        result: list[str] = []

        named_declarations = {
            "function_declaration",
            "class_declaration",
            "interface_declaration",
            "type_alias_declaration",
        }

        for child in node.named_children:

            if child.type in named_declarations:

                name_node = child.child_by_field_name("name")

                if name_node is not None:

                    result.append(
                        self._node_text(
                            node=name_node,
                            source=source,
                        ).strip()
                    )

                continue

            if child.type in {
                "lexical_declaration",
                "variable_declaration",
            }:

                declarators: list[Node] = []

                self._collect_nodes(
                    node=child,
                    node_type=("variable_declarator"),
                    result=declarators,
                )

                for declarator in declarators:

                    name_node = declarator.child_by_field_name("name")

                    if name_node is not None and (name_node.type == "identifier"):

                        result.append(
                            self._node_text(
                                node=name_node,
                                source=source,
                            ).strip()
                        )

        return result

    def _module_specifier(
        self,
        text: str,
    ) -> str | None:

        match = self._FROM_PATTERN.search(text)

        if match is None:
            return None

        return match.group("specifier")

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

        return source[node.start_byte : node.end_byte].decode("utf-8")
