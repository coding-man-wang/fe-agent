from __future__ import annotations

from collections import defaultdict

from agent_core.code_intelligence.definition_resolver import (
    SymbolDefinitionResolver,
)
from agent_core.code_intelligence.model import (
    CodeSymbol,
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
    Lexical Scope
        ↓
    Import / Namespace Binding
        ↓
    Definition Resolution
        ↓
    Resolved Reference
    """

    _VALUE_KINDS = {
        "function",
        "variable",
        "class",
        "component",
    }

    _CALLABLE_KINDS = {
        "function",
        "variable",
    }

    _TYPE_KINDS = {
        "type",
        "interface",
        "class",
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

    # ==================================
    # Public Resolution
    # ==================================

    def resolve(
        self,
        occurrence: SymbolReferenceOccurrence,
    ) -> ResolvedSymbolReference:

        owner = self.symbol_index.find_innermost_containing(
            relative_path=(occurrence.source_path),
            line_number=(occurrence.line_number),
        )

        source_symbol_id = owner.symbol_id if owner is not None else None

        if occurrence.target_form == "identifier":
            return self._resolve_identifier(
                occurrence=occurrence,
                owner=owner,
                source_symbol_id=(source_symbol_id),
            )

        if occurrence.target_form == "member":
            return self._resolve_member(
                occurrence=occurrence,
                owner=owner,
                source_symbol_id=(source_symbol_id),
            )

        return self._result(
            occurrence=occurrence,
            source_symbol_id=(source_symbol_id),
            status="unsupported",
            reason=("unknown_reference_target_form"),
        )

    # ==================================
    # Identifier
    # ==================================

    def _resolve_identifier(
        self,
        *,
        occurrence: SymbolReferenceOccurrence,
        owner: CodeSymbol | None,
        source_symbol_id: str | None,
    ) -> ResolvedSymbolReference:

        name = occurrence.name

        if not name:
            return self._result(
                occurrence=occurrence,
                source_symbol_id=(source_symbol_id),
                status="unsupported",
                reason="missing_identifier_name",
            )

        # ----------------------------------
        # Lexical Scope 优先
        # ----------------------------------

        local = self._resolve_local(
            source_path=(occurrence.source_path),
            name=name,
            owner=owner,
            reference_kind=(occurrence.reference_kind),
        )

        if local:

            return self._result(
                occurrence=occurrence,
                source_symbol_id=(source_symbol_id),
                status="resolved",
                target_symbol_ids=[item.symbol_id for item in local],
                reason=("local_lexical_binding"),
            )

        # ----------------------------------
        # Import Binding
        # ----------------------------------

        imports = self._imports_by_file_and_name.get(
            (
                occurrence.source_path,
                name,
            ),
            [],
        )

        if len(imports) > 1:

            return self._result(
                occurrence=occurrence,
                source_symbol_id=(source_symbol_id),
                status="ambiguous",
                reason=("multiple_import_bindings_" "with_same_local_name"),
            )

        if len(imports) == 1:

            binding = imports[0]

            # namespace 本身作为 identifier：
            #
            # const x = api
            #
            # 当前只知道它代表 Module，
            # 不是单一 CodeSymbol。
            if binding.binding_kind == "namespace":
                return self._result(
                    occurrence=occurrence,
                    source_symbol_id=(source_symbol_id),
                    status="unsupported",
                    reason=("namespace_binding_" "is_module_not_symbol"),
                )

            definition = self.definition_resolver.resolve(binding)

            return self._map_definition(
                occurrence=occurrence,
                source_symbol_id=(source_symbol_id),
                definition=definition,
            )

        return self._result(
            occurrence=occurrence,
            source_symbol_id=(source_symbol_id),
            status="unresolved",
            reason="no_visible_binding",
        )

    # ==================================
    # Member
    # ==================================

    def _resolve_member(
        self,
        *,
        occurrence: SymbolReferenceOccurrence,
        owner: CodeSymbol | None,
        source_symbol_id: str | None,
    ) -> ResolvedSymbolReference:

        member_name = occurrence.member_name

        receiver_name = occurrence.receiver_name

        if not member_name:

            return self._result(
                occurrence=occurrence,
                source_symbol_id=(source_symbol_id),
                status="unsupported",
                reason=("missing_member_name"),
            )

        # ==================================
        # this.method()
        # ==================================

        if receiver_name == "this":

            targets = self._resolve_this_member(
                source_path=(occurrence.source_path),
                member_name=member_name,
                owner=owner,
                reference_kind=(occurrence.reference_kind),
            )

            if targets:

                return self._result(
                    occurrence=occurrence,
                    source_symbol_id=(source_symbol_id),
                    status="resolved",
                    target_symbol_ids=[item.symbol_id for item in targets],
                    reason="this_member",
                )

            return self._result(
                occurrence=occurrence,
                source_symbol_id=(source_symbol_id),
                status="unresolved",
                reason=("this_member_not_found"),
            )

        # ==================================
        # api.listOrders()
        #
        # api 必须是 namespace import。
        # ==================================

        if receiver_name:

            # 同名局部对象会遮蔽 namespace import；不能沿旧导入生成错误图边。
            local_receiver = self._resolve_local(
                source_path=occurrence.source_path,
                name=receiver_name,
                owner=owner,
                reference_kind="read",
            )
            if local_receiver:
                return self._result(
                    occurrence=occurrence,
                    source_symbol_id=source_symbol_id,
                    status="unsupported",
                    reason="local_object_member_requires_value_flow",
                )

            imports = self._imports_by_file_and_name.get(
                (
                    occurrence.source_path,
                    receiver_name,
                ),
                [],
            )

            if len(imports) > 1:

                return self._result(
                    occurrence=occurrence,
                    source_symbol_id=(source_symbol_id),
                    status="ambiguous",
                    reason=("multiple_receiver_" "import_bindings"),
                )

            if len(imports) == 1:

                binding = imports[0]

                if binding.binding_kind != "namespace":

                    return self._result(
                        occurrence=occurrence,
                        source_symbol_id=(source_symbol_id),
                        status="unsupported",
                        reason=("member_receiver_import_" "is_not_namespace"),
                    )

                definition = self.definition_resolver.resolve_namespace_member(
                    binding=binding,
                    member_name=member_name,
                )

                return self._map_definition(
                    occurrence=occurrence,
                    source_symbol_id=(source_symbol_id),
                    definition=definition,
                )

            # --------------------------------
            # local object:
            #
            # store.fetchOrders()
            #
            # 需要 Value Flow / Type Analysis。
            # 不能按方法名全仓猜。
            # --------------------------------

            return self._result(
                occurrence=occurrence,
                source_symbol_id=(source_symbol_id),
                status="unsupported",
                reason=("local_object_member_" "requires_value_flow"),
            )

        return self._result(
            occurrence=occurrence,
            source_symbol_id=(source_symbol_id),
            status="unsupported",
            reason=("complex_member_receiver_" "requires_value_flow"),
        )

    # ==================================
    # this.member
    # ==================================

    def _resolve_this_member(
        self,
        *,
        source_path: str,
        member_name: str,
        owner: CodeSymbol | None,
        reference_kind: str,
    ) -> list[CodeSymbol]:

        if owner is None:
            return []

        scope = [
            owner,
            *self.symbol_index.parent_chain(owner.symbol_id),
        ]

        owner_class = next(
            (symbol for symbol in scope if symbol.kind == "class"),
            None,
        )

        if owner_class is None:
            return []

        candidates = self.symbol_index.find_by_file_and_name(
            relative_path=(source_path),
            name=member_name,
        )

        if reference_kind == "call":

            allowed_kinds = {
                "method",
            }

        elif reference_kind == "read":

            allowed_kinds = {
                "method",
                "variable",
            }

        else:
            return []

        return [
            item
            for item in candidates
            if (
                item.parent_symbol_id == owner_class.symbol_id
                and item.kind in allowed_kinds
            )
        ]

    # ==================================
    # Local Lexical Binding
    # ==================================

    def _resolve_local(
        self,
        *,
        source_path: str,
        name: str,
        owner: CodeSymbol | None,
        reference_kind: str,
    ) -> list[CodeSymbol]:

        allowed_kinds = self._allowed_local_kinds(reference_kind)

        candidates = [
            symbol
            for symbol in (
                self.symbol_index.find_by_file_and_name(
                    relative_path=(source_path),
                    name=name,
                )
            )
            if (symbol.kind in allowed_kinds)
        ]

        if not candidates:
            return []

        if owner is None:

            return [item for item in candidates if (item.parent_symbol_id is None)]

        # Recursive self reference
        self_reference = [
            item for item in candidates if (item.symbol_id == owner.symbol_id)
        ]

        if self_reference:
            return self_reference

        scope_ids: list[str | None] = [
            owner.symbol_id,
        ]

        parent_chain = self.symbol_index.parent_chain(owner.symbol_id)

        scope_ids.extend(item.symbol_id for item in parent_chain)

        # TypeScript module scope
        scope_ids.append(None)

        ranked: list[
            tuple[
                int,
                CodeSymbol,
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

    def _allowed_local_kinds(
        self,
        reference_kind: str,
    ) -> set[str]:

        if reference_kind == "call":
            return self._CALLABLE_KINDS

        if reference_kind == "read":
            return self._VALUE_KINDS

        if reference_kind == "type_reference":
            return self._TYPE_KINDS

        return set()

    # ==================================
    # Definition → Reference Result
    # ==================================

    def _map_definition(
        self,
        *,
        occurrence: SymbolReferenceOccurrence,
        source_symbol_id: str | None,
        definition,
    ) -> ResolvedSymbolReference:

        if definition.status == "resolved":

            return self._result(
                occurrence=occurrence,
                source_symbol_id=(source_symbol_id),
                status="resolved",
                target_symbol_ids=(definition.target_symbol_ids),
                reason="import_binding",
            )

        if definition.status == "external_module":

            return self._result(
                occurrence=occurrence,
                source_symbol_id=(source_symbol_id),
                status="external",
                reason=("external_import_binding"),
            )

        if definition.status == "ambiguous_export":

            return self._result(
                occurrence=occurrence,
                source_symbol_id=(source_symbol_id),
                status="ambiguous",
                reason=(definition.reason),
            )

        return self._result(
            occurrence=occurrence,
            source_symbol_id=(source_symbol_id),
            status="unresolved",
            reason=("definition_" + definition.status),
        )

    @staticmethod
    def _result(
        *,
        occurrence: SymbolReferenceOccurrence,
        source_symbol_id: str | None,
        status: str,
        target_symbol_ids: list[str] | None = None,
        reason: str | None = None,
    ) -> ResolvedSymbolReference:

        return ResolvedSymbolReference(
            source_path=(occurrence.source_path),
            reference_kind=(occurrence.reference_kind),
            expression_text=(occurrence.expression_text),
            line_number=(occurrence.line_number),
            source_symbol_id=(source_symbol_id),
            status=status,
            target_symbol_ids=(target_symbol_ids or []),
            reason=reason,
        )
