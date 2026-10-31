from agent_core.code_intelligence.adapters.typescript import (
    TypeScriptSymbolExtractorAdapter,
)
from agent_core.code_intelligence.adapters.typescript_imports import (
    TypeScriptImportBindingExtractorAdapter,
)
from agent_core.code_intelligence.definition_resolver import (
    SymbolDefinitionResolver,
)
from agent_core.code_intelligence.model import (
    SourceDocument,
    TechnologyProfile,
)
from agent_core.code_intelligence.symbol_index import (
    SymbolIndex,
)


def make_profile():
    return TechnologyProfile(
        languages=[
            "typescript",
        ],
        frameworks=[
            "vue",
        ],
    )


def test_exported_class_method_is_not_module_export():

    adapter = TypeScriptSymbolExtractorAdapter()

    profile = make_profile()

    result = adapter.extract_symbols(
        profile=profile,
        document=SourceDocument(
            relative_path=("src/api/orders.ts"),
            language="typescript",
            content="""
export class OrderService {
  listOrders() {
    return []
  }
}
""".strip(),
        ),
    )

    by_name = {item.name: item for item in result.symbols}

    assert by_name["OrderService"].exported is True

    assert by_name["listOrders"].exported is False


def test_named_import_binding_is_extracted():

    adapter = TypeScriptImportBindingExtractorAdapter()

    result = adapter.extract_import_bindings(
        profile=make_profile(),
        document=SourceDocument(
            relative_path=("src/views/orders.ts"),
            language="typescript",
            content="""
import {
  listOrders as fetchOrders
} from "@/api/orders"
""".strip(),
        ),
    )

    assert len(result.bindings) == 1

    binding = result.bindings[0]

    assert binding.imported_name == "listOrders"

    assert binding.local_name == "fetchOrders"

    assert binding.module_specifier == "@/api/orders"


def test_type_only_import_is_preserved():

    adapter = TypeScriptImportBindingExtractorAdapter()

    result = adapter.extract_import_bindings(
        profile=make_profile(),
        document=SourceDocument(
            relative_path=("src/api/client.ts"),
            language="typescript",
            content="""
import type {
  Order
} from "@/types/domain"
""".strip(),
        ),
    )

    assert result.bindings[0].type_only is True


def test_alias_import_resolves_to_exported_symbol():

    profile = make_profile()

    symbol_adapter = TypeScriptSymbolExtractorAdapter()

    import_adapter = TypeScriptImportBindingExtractorAdapter()

    orders_result = symbol_adapter.extract_symbols(
        profile=profile,
        document=SourceDocument(
            relative_path=("src/api/orders.ts"),
            language="typescript",
            content="""
export function listOrders() {
  return []
}
""".strip(),
        ),
    )

    consumer = SourceDocument(
        relative_path=("src/views/orders.ts"),
        language="typescript",
        content="""
import {
  listOrders as fetchOrders
} from "@/api/orders"
""".strip(),
    )

    imports = import_adapter.extract_import_bindings(
        profile=profile,
        document=consumer,
    )

    index = SymbolIndex.build(
        [
            orders_result,
        ]
    )

    resolver = SymbolDefinitionResolver(
        symbol_index=index,
        file_paths={
            "src/api/orders.ts",
            "src/views/orders.ts",
        },
    )

    resolution = resolver.resolve(imports.bindings[0])

    assert resolution.status == "resolved"

    assert resolution.target_path == "src/api/orders.ts"

    assert resolution.target_symbol_ids == [
        "src/api/orders.ts" "::function" "::listOrders"
    ]


def test_external_package_is_not_treated_as_repository_symbol():

    adapter = TypeScriptImportBindingExtractorAdapter()

    result = adapter.extract_import_bindings(
        profile=make_profile(),
        document=SourceDocument(
            relative_path=("src/main.ts"),
            language="typescript",
            content="""
import {
  ref
} from "vue"
""".strip(),
        ),
    )

    resolver = SymbolDefinitionResolver(
        symbol_index=(SymbolIndex()),
        file_paths={
            "src/main.ts",
        },
    )

    resolution = resolver.resolve(result.bindings[0])

    assert resolution.status == "external_module"


def test_missing_internal_module_fails_closed():

    adapter = TypeScriptImportBindingExtractorAdapter()

    result = adapter.extract_import_bindings(
        profile=make_profile(),
        document=SourceDocument(
            relative_path=("src/main.ts"),
            language="typescript",
            content="""
import {
  missing
} from "@/api/missing"
""".strip(),
        ),
    )

    resolver = SymbolDefinitionResolver(
        symbol_index=(SymbolIndex()),
        file_paths={
            "src/main.ts",
        },
    )

    resolution = resolver.resolve(result.bindings[0])

    assert resolution.status == "unresolved_module"


def test_vue_import_resolves_to_typescript_definition():
    profile = make_profile()

    ts_adapter = TypeScriptSymbolExtractorAdapter()

    vue_adapter = VueSfcSymbolExtractorAdapter()

    orders_result = ts_adapter.extract_symbols(
        profile=profile,
        document=SourceDocument(
            relative_path=("src/api/orders.ts"),
            language="typescript",
            content="""
export function listOrders() {
  return []
}
""".strip(),
        ),
    )

    vue_document = SourceDocument(
        relative_path=("src/views/OrdersView.vue"),
        content="""
<script setup lang="ts">

import {
  listOrders
} from "@/api/orders"

async function loadOrders() {
  return listOrders()
}

</script>
""".strip(),
    )

    imports = vue_adapter.extract_import_bindings(
        profile=profile,
        document=vue_document,
    )

    index = SymbolIndex.build(
        [
            orders_result,
        ]
    )

    resolver = SymbolDefinitionResolver(
        symbol_index=index,
        file_paths={
            "src/api/orders.ts",
            ("src/views/" "OrdersView.vue"),
        },
    )

    resolution = resolver.resolve(
        next(item for item in imports.bindings if (item.local_name == "listOrders"))
    )

    assert resolution.status == "resolved"

    assert resolution.target_path == "src/api/orders.ts"
