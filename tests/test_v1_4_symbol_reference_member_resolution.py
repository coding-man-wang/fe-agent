from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from agent_core.code_intelligence.adapters.typescript import (
    TypeScriptSymbolExtractorAdapter,
)
from agent_core.code_intelligence.adapters.typescript_exports import (
    TypeScriptModuleExportExtractor,
)
from agent_core.code_intelligence.adapters.typescript_imports import (
    TypeScriptImportBindingExtractorAdapter,
)
from agent_core.code_intelligence.adapters.typescript_references import (
    TypeScriptSymbolReferenceExtractor,
)
from agent_core.code_intelligence.adapters.vue import (
    VueSfcSymbolExtractorAdapter,
)
from agent_core.code_intelligence.definition_resolver import (
    SymbolDefinitionResolver,
)
from agent_core.code_intelligence.model import (
    ImportBinding,
    SourceDocument,
    SymbolReferenceOccurrence,
    TechnologyProfile,
)
from agent_core.code_intelligence.module_export import (
    ModuleExportTable,
)
from agent_core.code_intelligence.reference_resolver import (
    SymbolReferenceResolver,
)
from agent_core.code_intelligence.symbol_graph import (
    SymbolReferenceGraphBuilder,
)
from agent_core.code_intelligence.symbol_index import (
    SymbolIndex,
)

PROFILE = TechnologyProfile(
    languages=[
        "typescript",
    ],
    frameworks=[
        "vue",
    ],
)


@dataclass
class RepositoryFixture:
    symbol_index: SymbolIndex
    import_bindings: list[ImportBinding]
    definition_resolver: SymbolDefinitionResolver
    reference_resolver: SymbolReferenceResolver


def document(
    path: str,
    content: str,
) -> SourceDocument:

    language = None

    suffix = PurePosixPath(path).suffix.lower()

    if suffix == ".ts":
        language = "typescript"

    elif suffix == ".tsx":
        language = "tsx"

    return SourceDocument(
        relative_path=path,
        content=content.strip(),
        language=language,
    )


def build_repository(
    documents: list[SourceDocument],
) -> RepositoryFixture:

    ts_symbol = TypeScriptSymbolExtractorAdapter()

    ts_import = TypeScriptImportBindingExtractorAdapter()

    ts_export = TypeScriptModuleExportExtractor()

    vue = VueSfcSymbolExtractorAdapter()

    symbol_results = []

    import_bindings: list[ImportBinding] = []

    export_results = []

    for item in documents:

        if item.relative_path.endswith(".vue"):

            symbol_results.append(
                vue.extract_symbols(
                    profile=PROFILE,
                    document=item,
                )
            )

            import_result = vue.extract_import_bindings(
                profile=PROFILE,
                document=item,
            )

            export_results.append(
                vue.extract_module_exports(
                    profile=PROFILE,
                    document=item,
                )
            )

        else:

            symbol_results.append(
                ts_symbol.extract_symbols(
                    profile=PROFILE,
                    document=item,
                )
            )

            import_result = ts_import.extract_import_bindings(
                profile=PROFILE,
                document=item,
            )

            export_results.append(
                ts_export.extract_module_exports(
                    profile=PROFILE,
                    document=item,
                )
            )

        import_bindings.extend(import_result.bindings)

    symbol_index = SymbolIndex.build(symbol_results)

    export_table = ModuleExportTable.build(export_results)

    file_paths = {item.relative_path for item in documents}

    definition_resolver = SymbolDefinitionResolver(
        symbol_index=(symbol_index),
        export_table=(export_table),
        file_paths=(file_paths),
    )

    reference_resolver = SymbolReferenceResolver(
        symbol_index=(symbol_index),
        definition_resolver=(definition_resolver),
        import_bindings=(import_bindings),
    )

    return RepositoryFixture(
        symbol_index=symbol_index,
        import_bindings=(import_bindings),
        definition_resolver=(definition_resolver),
        reference_resolver=(reference_resolver),
    )


def extract_references(
    item: SourceDocument,
):

    if item.relative_path.endswith(".vue"):

        return VueSfcSymbolExtractorAdapter().extract_references(
            profile=PROFILE,
            document=item,
        )

    return TypeScriptSymbolReferenceExtractor().extract_references(
        profile=PROFILE,
        document=item,
    )


def resolve_all(
    *,
    repository: RepositoryFixture,
    item: SourceDocument,
):

    extracted = extract_references(item)

    return [
        repository.reference_resolver.resolve(reference)
        for reference in extracted.references
    ]


# ==================================
# 1. Namespace Member Call
# ==================================


def test_namespace_import_member_call_resolves_through_barrel():

    orders = document(
        "src/api/orders.ts",
        """
export function listOrders() {
  return []
}
""",
    )

    barrel = document(
        "src/api/index.ts",
        """
export * from "./orders"
""",
    )

    consumer = document(
        "src/views/orders.ts",
        """
import * as api from "@/api"

export function loadOrders() {
  return api.listOrders()
}
""",
    )

    repository = build_repository(
        [
            orders,
            barrel,
            consumer,
        ]
    )

    extracted = extract_references(consumer)

    member_call = next(
        item
        for item in extracted.references
        if (item.reference_kind == "call" and item.expression_text == "api.listOrders")
    )

    assert member_call.target_form == "member"

    assert member_call.receiver_name == "api"

    assert member_call.member_name == "listOrders"

    resolution = repository.reference_resolver.resolve(member_call)

    assert resolution.status == "resolved"

    assert resolution.source_symbol_id == (
        "src/views/orders.ts" "::function" "::loadOrders"
    )

    assert resolution.target_symbol_ids == [
        "src/api/orders.ts" "::function" "::listOrders"
    ]


# ==================================
# 2. this.method()
# ==================================


def test_this_method_call_resolves_to_class_method():

    service = document(
        "src/services/orderService.ts",
        """
export class OrderService {
  loadOrders() {
    return this.normalizeStatus()
  }

  normalizeStatus() {
    return "all"
  }
}
""",
    )

    repository = build_repository(
        [
            service,
        ]
    )

    extracted = extract_references(service)

    member_call = next(
        item
        for item in extracted.references
        if (item.expression_text == "this.normalizeStatus")
    )

    assert member_call.receiver_name == "this"

    resolution = repository.reference_resolver.resolve(member_call)

    assert resolution.status == "resolved"

    assert resolution.source_symbol_id == (
        "src/services/" "orderService.ts" "::method" "::OrderService.loadOrders"
    )

    assert resolution.target_symbol_ids == [
        "src/services/" "orderService.ts" "::method" "::OrderService" ".normalizeStatus"
    ]


# ==================================
# 3. Generic Object Member Fail Closed
# ==================================


def test_generic_object_member_call_requires_value_flow():

    source = document(
        "src/stores/consumer.ts",
        """
export function loadOrders(store: any) {
  return store.fetchOrders()
}
""",
    )

    repository = build_repository(
        [
            source,
        ]
    )

    extracted = extract_references(source)

    member_call = next(
        item
        for item in extracted.references
        if (item.expression_text == "store.fetchOrders")
    )

    resolution = repository.reference_resolver.resolve(member_call)

    assert resolution.status == "unsupported"

    assert resolution.reason == ("local_object_member_" "requires_value_flow")

    assert resolution.target_symbol_ids == []


# ==================================
# 4. Imported Value Read
# ==================================


def test_imported_value_read_resolves():

    constants = document(
        "src/constants/orders.ts",
        """
export const DEFAULT_STATUS = "all"
""",
    )

    consumer = document(
        "src/views/orders.ts",
        """
import {
  DEFAULT_STATUS
} from "@/constants/orders"

export function initialStatus() {
  return DEFAULT_STATUS
}
""",
    )

    repository = build_repository(
        [
            constants,
            consumer,
        ]
    )

    extracted = extract_references(consumer)

    reference = next(
        item
        for item in extracted.references
        if (item.reference_kind == "read" and item.name == "DEFAULT_STATUS")
    )

    resolution = repository.reference_resolver.resolve(reference)

    assert resolution.status == "resolved"

    assert resolution.target_symbol_ids == [
        "src/constants/orders.ts" "::variable" "::DEFAULT_STATUS"
    ]


# ==================================
# 5. Lexical Read Shadowing
# ==================================


def test_local_value_read_prefers_nearest_scope():

    source = document(
        "src/views/orders.ts",
        """
const status = "outer"

export function loadOrders() {
  const status = "inner"

  return status
}
""",
    )

    repository = build_repository(
        [
            source,
        ]
    )

    extracted = extract_references(source)

    reference = next(
        item
        for item in extracted.references
        if (
            item.reference_kind == "read"
            and item.name == "status"
            and item.line_number == 6
        )
    )

    resolution = repository.reference_resolver.resolve(reference)

    assert resolution.status == "resolved"

    assert resolution.target_symbol_ids == [
        "src/views/orders.ts" "::variable" "::loadOrders.status"
    ]


# ==================================
# 6. Imported Type Reference
# ==================================


def test_imported_type_reference_resolves():

    domain = document(
        "src/types/order.ts",
        """
export interface Order {
  id: string
}
""",
    )

    consumer = document(
        "src/api/orders.ts",
        """
import type {
  Order
} from "@/types/order"

export function identity(
  order: Order
): Order {
  return order
}
""",
    )

    repository = build_repository(
        [
            domain,
            consumer,
        ]
    )

    extracted = extract_references(consumer)

    type_references = [
        item
        for item in extracted.references
        if (item.reference_kind == "type_reference" and item.name == "Order")
    ]

    assert len(type_references) >= 1

    for reference in type_references:

        resolution = repository.reference_resolver.resolve(reference)

        assert resolution.status == "resolved"

        assert resolution.target_symbol_ids == [
            "src/types/order.ts" "::interface" "::Order"
        ]


# ==================================
# 7. Namespace Member Read
# ==================================


def test_namespace_member_read_resolves():

    orders = document(
        "src/api/orders.ts",
        """
export function listOrders() {
  return []
}
""",
    )

    consumer = document(
        "src/views/orders.ts",
        """
import * as api from "@/api/orders"

export const loader = api.listOrders
""",
    )

    repository = build_repository(
        [
            orders,
            consumer,
        ]
    )

    extracted = extract_references(consumer)

    reference = next(
        item
        for item in extracted.references
        if (item.reference_kind == "read" and item.expression_text == "api.listOrders")
    )

    resolution = repository.reference_resolver.resolve(reference)

    assert resolution.status == "resolved"

    assert resolution.target_symbol_ids == [
        "src/api/orders.ts" "::function" "::listOrders"
    ]


# ==================================
# 8. Default Import Regression
# ==================================


def test_default_import_resolves_after_export_table_integration():

    client = document(
        "src/api/client.ts",
        """
export default function createClient() {
  return {}
}
""",
    )

    consumer = document(
        "src/main.ts",
        """
import createClient from "@/api/client"

export function boot() {
  return createClient()
}
""",
    )

    repository = build_repository(
        [
            client,
            consumer,
        ]
    )

    resolutions = resolve_all(
        repository=repository,
        item=consumer,
    )

    call = next(
        item
        for item in resolutions
        if (item.reference_kind == "call" and item.expression_text == "createClient")
    )

    assert call.status == "resolved"

    assert call.target_symbol_ids == ["src/api/client.ts" "::function" "::createClient"]


# ==================================
# 9. General Symbol Graph
# ==================================


def test_general_reference_graph_contains_calls_and_references():

    source = document(
        "src/orders.ts",
        """
const DEFAULT_STATUS = "all"

function normalizeStatus() {
  return DEFAULT_STATUS
}

export function loadOrders() {
  return normalizeStatus()
}
""",
    )

    repository = build_repository(
        [
            source,
        ]
    )

    resolutions = resolve_all(
        repository=repository,
        item=source,
    )

    graph = SymbolReferenceGraphBuilder().build(resolutions)

    load_id = "src/orders.ts" "::function" "::loadOrders"

    normalize_id = "src/orders.ts" "::function" "::normalizeStatus"

    default_id = "src/orders.ts" "::variable" "::DEFAULT_STATUS"

    load_edges = graph.outgoing(load_id)

    assert any(
        edge.relation == "calls" and (edge.target_symbol_id == normalize_id)
        for edge in load_edges
    )

    normalize_edges = graph.outgoing(normalize_id)

    assert any(
        edge.relation == "references" and (edge.target_symbol_id == default_id)
        for edge in normalize_edges
    )


# ==================================
# 10. Vue Reference Mapping
# ==================================


def test_vue_member_and_identifier_references_use_real_repository_path():

    orders = document(
        "src/api/orders.ts",
        """
export function listOrders() {
  return []
}
""",
    )

    barrel = document(
        "src/api/index.ts",
        """
export * from "./orders"
""",
    )

    view = document(
        "src/views/OrdersView.vue",
        """
<template>
  <main>Orders</main>
</template>

<script setup lang="ts">
import * as api from "@/api"

async function loadOrders() {
  return api.listOrders()
}
</script>
""",
    )

    repository = build_repository(
        [
            orders,
            barrel,
            view,
        ]
    )

    extracted = extract_references(view)

    call = next(
        item
        for item in extracted.references
        if (item.reference_kind == "call" and item.expression_text == "api.listOrders")
    )

    assert call.source_path == ("src/views/" "OrdersView.vue")

    assert ".__reference_" not in call.source_path

    resolution = repository.reference_resolver.resolve(call)

    assert resolution.status == "resolved"

    assert resolution.source_symbol_id == (
        "src/views/" "OrdersView.vue" "::function" "::script_setup.loadOrders"
    )

    assert resolution.target_symbol_ids == [
        "src/api/orders.ts" "::function" "::listOrders"
    ]


# ==================================
# 11. Architecture Boundary Regression
# ==================================


def test_reference_resolver_has_no_vue_extraction_responsibility():

    assert not hasattr(
        SymbolReferenceResolver,
        "extract_references",
    )


# ==================================
# 12. Member Occurrence Contract
# ==================================


def test_member_occurrence_contains_receiver_and_member_name():

    source = document(
        "src/api/consumer.ts",
        """
function run(api: any) {
  return api.listOrders()
}
""",
    )

    extracted = extract_references(source)

    call = next(
        item
        for item in extracted.references
        if (item.expression_text == "api.listOrders")
    )

    assert call.target_form == "member"

    assert call.receiver_name == "api"

    assert call.member_name == "listOrders"


def test_local_receiver_shadows_namespace_import_without_false_graph_edge():
    orders = document("src/api/orders.ts", "export function listOrders() { return [] }")
    consumer = document(
        "src/views/orders.ts",
        """
import * as api from "@/api/orders"
export function loadOrders() {
  const api = makeLocalApi()
  return api.listOrders()
}
""",
    )
    repository = build_repository([orders, consumer])
    resolutions = resolve_all(repository=repository, item=consumer)
    call = next(
        item for item in resolutions if item.expression_text == "api.listOrders"
    )
    assert call.status == "unsupported"
    assert call.target_symbol_ids == []
    graph = SymbolReferenceGraphBuilder().build(resolutions)
    assert not graph.incoming("src/api/orders.ts::function::listOrders")
