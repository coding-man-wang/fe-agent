from __future__ import annotations

from collections import defaultdict

from agent_core.code_intelligence.definition_resolver import (
    SymbolDefinitionResolver,
)
from agent_core.code_intelligence.model import (
    ImportBinding,
    ResolvedSymbolReference,
    SymbolReferenceOccurrence,
)
from agent_core.code_intelligence.symbol_index import (
    SymbolIndex,
)


class SymbolReferenceResolver:
    """
    Reference Occurrence
        ↓
    Lexical Binding
        ↓
    Import Binding
        ↓
    Definition Resolution
        ↓
    Resolved Symbol Reference
    """

    _DIRECT_CALLABLE_KINDS = {
        "function",
        "variable",
    }

    def __init__(
        self,
        *,
        symbol_index: SymbolIndex,
        definition_resolver: SymbolDefinitionResolver,
        import_bindings: list[ImportBinding],
    ) -> None:

        self.symbol_index = symbol_index

        self.definition_resolver = definition_resolver

        self._imports_by_file_and_name: dict[
            tuple[str, str],
            list[ImportBinding],
        ] = defaultdict(list)

        for binding in import_bindings:

            if not binding.local_name:
                continue

            self._imports_by_file_and_name[
                (
                    binding.source_path,
                    binding.local_name,
                )
            ].append(binding)

    def resolve(
        self,
        occurrence: SymbolReferenceOccurrence,
    ) -> ResolvedSymbolReference:

        owner = self.symbol_index.find_innermost_containing(
            relative_path=(occurrence.source_path),
            line_number=(occurrence.line_number),
        )

        source_symbol_id = owner.symbol_id if owner is not None else None

        # ==================================
        # 4A 只解析 identifier call。
        # ==================================

        if (
            occurrence.reference_kind != "call"
            or occurrence.target_form != "identifier"
            or not occurrence.name
        ):

            return ResolvedSymbolReference(
                source_path=(occurrence.source_path),
                reference_kind=(occurrence.reference_kind),
                expression_text=(occurrence.expression_text),
                line_number=(occurrence.line_number),
                source_symbol_id=(source_symbol_id),
                status="unsupported",
                reason=("member_or_non_identifier_" "reference_deferred_to_" "v1_4_4b"),
            )

        name = occurrence.name

        # ==================================
        # Local Lexical Binding 优先
        #
        # 例如：
        #
        # import { listOrders } ...
        #
        # function load() {
        #   const listOrders = () => []
        #   listOrders()
        # }
        #
        # 必须解析到 Local Binding，
        # 不能解析到 import。
        # ==================================

        local = self._resolve_local(
            source_path=(occurrence.source_path),
            name=name,
            owner=owner,
        )

        if local:

            return ResolvedSymbolReference(
                source_path=(occurrence.source_path),
                reference_kind="call",
                expression_text=(occurrence.expression_text),
                line_number=(occurrence.line_number),
                source_symbol_id=(source_symbol_id),
                status="resolved",
                target_symbol_ids=[symbol.symbol_id for symbol in local],
                reason="local_lexical_binding",
            )

        # ==================================
        # Imported Binding
        # ==================================

        import_bindings = self._imports_by_file_and_name.get(
            (
                occurrence.source_path,
                name,
            ),
            [],
        )

        if len(import_bindings) > 1:

            return ResolvedSymbolReference(
                source_path=(occurrence.source_path),
                reference_kind="call",
                expression_text=(occurrence.expression_text),
                line_number=(occurrence.line_number),
                source_symbol_id=(source_symbol_id),
                status="ambiguous",
                reason=("multiple_import_bindings_" "with_same_local_name"),
            )

        if len(import_bindings) == 1:

            definition = self.definition_resolver.resolve(import_bindings[0])

            if definition.status == "resolved":

                return ResolvedSymbolReference(
                    source_path=(occurrence.source_path),
                    reference_kind="call",
                    expression_text=(occurrence.expression_text),
                    line_number=(occurrence.line_number),
                    source_symbol_id=(source_symbol_id),
                    status="resolved",
                    target_symbol_ids=(definition.target_symbol_ids),
                    reason=("import_binding"),
                )

            if definition.status == "external_module":

                return ResolvedSymbolReference(
                    source_path=(occurrence.source_path),
                    reference_kind="call",
                    expression_text=(occurrence.expression_text),
                    line_number=(occurrence.line_number),
                    source_symbol_id=(source_symbol_id),
                    status="external",
                    reason=("external_import_binding"),
                )

            if definition.status == "ambiguous_export":

                return ResolvedSymbolReference(
                    source_path=(occurrence.source_path),
                    reference_kind="call",
                    expression_text=(occurrence.expression_text),
                    line_number=(occurrence.line_number),
                    source_symbol_id=(source_symbol_id),
                    status="ambiguous",
                    reason=(definition.reason),
                )

            return ResolvedSymbolReference(
                source_path=(occurrence.source_path),
                reference_kind="call",
                expression_text=(occurrence.expression_text),
                line_number=(occurrence.line_number),
                source_symbol_id=(source_symbol_id),
                status="unresolved",
                reason=("import_definition_" f"{definition.status}"),
            )

        # ==================================
        # 没有 Local / Import Binding。
        #
        # 不进行全仓名称搜索猜测。
        # ==================================

        return ResolvedSymbolReference(
            source_path=(occurrence.source_path),
            reference_kind="call",
            expression_text=(occurrence.expression_text),
            line_number=(occurrence.line_number),
            source_symbol_id=(source_symbol_id),
            status="unresolved",
            reason=("no_visible_binding"),
        )

    def _resolve_local(
        self,
        *,
        source_path: str,
        name: str,
        owner,
    ):
        candidates = [
            symbol
            for symbol in (
                self.symbol_index.find_by_file_and_name(
                    relative_path=(source_path),
                    name=name,
                )
            )
            if (symbol.kind in self._DIRECT_CALLABLE_KINDS)
        ]

        if not candidates:
            return []

        # ==================================
        # Module-level Call
        #
        # 当前没有 Module Symbol，
        # 所以只能看到 top-level definitions。
        # ==================================

        if owner is None:
            return [item for item in candidates if (item.parent_symbol_id is None)]

        # ==================================
        # Recursive Call
        # ==================================

        recursive = [item for item in candidates if (item.symbol_id == owner.symbol_id)]

        if recursive:
            return recursive

        # ==================================
        # Visible Scope Chain
        #
        # owner:
        #   inner()
        #
        # 可见 Container：
        #
        # inner
        # outer
        # component / module-level
        # ==================================

        scope_ids: list[str | None] = [
            owner.symbol_id,
        ]

        parent_chain = self.symbol_index.parent_chain(owner.symbol_id)

        scope_ids.extend(item.symbol_id for item in parent_chain)

        # TypeScript top-level Symbol
        # parent_symbol_id = None。
        scope_ids.append(None)

        ranked: list[
            tuple[
                int,
                object,
            ]
        ] = []

        for candidate in candidates:

            try:
                rank = scope_ids.index(candidate.parent_symbol_id)
            except ValueError:
                continue

            ranked.append(
                (
                    rank,
                    candidate,
                )
            )

        if not ranked:
            return []

        best_rank = min(item[0] for item in ranked)

        return [item[1] for item in ranked if item[0] == best_rank]

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
            raise ValueError("Vue Adapter 不支持该文件: " + document.relative_path)

        extractor = TypeScriptSymbolReferenceExtractor()

        references: list[SymbolReferenceOccurrence] = []

        issues: list[CodeIntelligenceIssue] = []

        blocks = self._extract_script_blocks(document.content)

        for block_index, block in enumerate(
            blocks,
            start=1,
        ):

            if block.has_src or block.language not in {
                "ts",
                "tsx",
            }:
                continue

            suffix = ".tsx" if block.language == "tsx" else ".ts"

            virtual_document = SourceDocument(
                relative_path=(
                    f"{document.relative_path}"
                    f".__reference_"
                    f"{block_index}"
                    f"{suffix}"
                ),
                content=block.content,
                language=("tsx" if block.language == "tsx" else "typescript"),
            )

            result = extractor.extract_references(
                profile=profile,
                document=virtual_document,
            )

            for reference in result.references:
                references.append(
                    reference.model_copy(
                        deep=True,
                        update={
                            "source_path": (document.relative_path),
                            "line_number": (reference.line_number + block.line_offset),
                        },
                    )
                )

            for issue in result.issues:
                issues.append(
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
                )

        return SymbolReferenceExtractionResult(
            adapter_id=self.adapter_id,
            relative_path=(document.relative_path),
            references=references,
            issues=issues,
        )
