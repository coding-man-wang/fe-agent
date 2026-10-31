from agent_core.code_intelligence.adapters.typescript import (
    TypeScriptSymbolExtractorAdapter,
)
from agent_core.code_intelligence.adapters.vue import (
    VueSfcSymbolExtractorAdapter,
)
from agent_core.code_intelligence.model import (
    SourceDocument,
    TechnologyProfile,
)
from agent_core.code_intelligence.registry import (
    SymbolExtractorRegistry,
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


def extract(
    content: str,
    *,
    relative_path: str = ("src/views/OrdersView.vue"),
):
    adapter = VueSfcSymbolExtractorAdapter()

    profile = make_profile()

    document = SourceDocument(
        relative_path=relative_path,
        content=content,
        language=None,
    )

    assert adapter.supports(
        profile=profile,
        document=document,
    )

    return adapter.extract_symbols(
        profile=profile,
        document=document,
    )


def test_vue_component_symbol_is_created():

    result = extract("""
<template>
  <main>Orders</main>
</template>

<script setup lang="ts">
const count = 1
</script>
""".strip())

    component = next(symbol for symbol in result.symbols if symbol.kind == "component")

    assert component.name == "OrdersView"

    assert component.relative_path == "src/views/OrdersView.vue"

    assert component.exported is True

    assert component.parent_symbol_id is None


def test_script_setup_typescript_symbols_are_extracted():

    result = extract("""
<template>
  <button @click="loadOrders">
    Load
  </button>
</template>

<script setup lang="ts">
const selectedStatus = "all"

function loadOrders() {
  return selectedStatus
}
</script>
""".strip())

    by_name = {symbol.name: symbol for symbol in result.symbols}

    assert by_name["selectedStatus"].kind == "variable"

    assert by_name["loadOrders"].kind == "function"

    assert by_name["loadOrders"].relative_path == "src/views/OrdersView.vue"

    assert ".__script_setup_" not in (by_name["loadOrders"].symbol_id)


def test_script_setup_symbols_are_children_of_component():

    result = extract("""
<script setup lang="ts">
function loadOrders() {
  return []
}
</script>
""".strip())

    by_name = {symbol.name: symbol for symbol in result.symbols}

    component = by_name["OrdersView"]

    load_orders = by_name["loadOrders"]

    assert load_orders.parent_symbol_id == component.symbol_id


def test_nested_function_keeps_nested_parent_relationship():

    result = extract("""
<script setup lang="ts">
function loadOrders() {
  function normalizeStatus() {
    return "all"
  }

  return normalizeStatus()
}
</script>
""".strip())

    by_name = {symbol.name: symbol for symbol in result.symbols}

    assert (
        by_name["normalizeStatus"].parent_symbol_id == by_name["loadOrders"].symbol_id
    )


def test_vue_line_numbers_map_back_to_original_sfc():

    result = extract("""
<template>
  <main>
    Orders
  </main>
</template>

<script setup lang="ts">
const selectedStatus = "all"

function loadOrders() {
  return []
}
</script>
""".strip())

    by_name = {symbol.name: symbol for symbol in result.symbols}

    # 原始 Vue 文件：
    #
    # 7: const selectedStatus...
    # 9: function loadOrders...
    #
    assert by_name["selectedStatus"].start_line == 7

    assert by_name["loadOrders"].start_line == 9


def test_normal_script_and_script_setup_can_coexist():

    result = extract("""
<script lang="ts">
export const MODULE_NAME = "orders"
</script>

<script setup lang="ts">
const selectedStatus = "all"

function loadOrders() {
  return []
}
</script>
""".strip())

    by_name = {symbol.name: symbol for symbol in result.symbols}

    assert by_name["MODULE_NAME"].exported is True

    assert by_name["selectedStatus"].exported is False

    assert by_name["loadOrders"].exported is False


def test_script_setup_arrow_function_is_function():

    result = extract("""
<script setup lang="ts">
const loadOrders = async () => {
  return []
}
</script>
""".strip())

    by_name = {symbol.name: symbol for symbol in result.symbols}

    assert by_name["loadOrders"].kind == "function"


def test_vue_tsx_script_uses_tsx_parser():

    result = extract("""
<script setup lang="tsx">
const renderOrders = () => {
  return <div>Orders</div>
}
</script>
""".strip())

    by_name = {symbol.name: symbol for symbol in result.symbols}

    assert by_name["renderOrders"].kind == "function"

    assert result.issues == []


def test_javascript_script_is_gracefully_skipped():

    result = extract("""
<script setup>
const count = 1
</script>
""".strip())

    names = {symbol.name for symbol in result.symbols}

    # Component 仍然存在。
    assert "OrdersView" in names

    # 当前 v1.4.2 尚未开启 JS Adapter。
    assert "count" not in names

    assert len(result.issues) == 1

    assert "暂未支持" in result.issues[0].message


def test_template_only_component_is_valid():

    result = extract("""
<template>
  <div>Static component</div>
</template>
""".strip())

    assert len(result.symbols) == 1

    assert result.symbols[0].kind == "component"

    assert result.issues == []


def test_typescript_syntax_error_keeps_vue_component_and_recoverable_symbols():

    result = extract("""
<script setup lang="ts">
function validFunction() {
  return 1
}

function broken( {
</script>
""".strip())

    names = {symbol.name for symbol in result.symbols}

    assert "OrdersView" in names

    assert "validFunction" in names

    assert len(result.issues) >= 1


def test_registry_resolves_vue_adapter():

    registry = SymbolExtractorRegistry()

    registry.register(TypeScriptSymbolExtractorAdapter())

    registry.register(VueSfcSymbolExtractorAdapter())

    profile = make_profile()

    document = SourceDocument(
        relative_path=("src/views/OrdersView.vue"),
        content=("<script setup " 'lang="ts">' "const count = 1" "</script>"),
    )

    resolved = registry.resolve(
        profile=profile,
        document=document,
    )

    assert resolved.adapter_id == "vue-sfc"


def test_vue_symbols_can_enter_common_symbol_index():

    result = extract("""
<script setup lang="ts">
function loadOrders() {
  return []
}
</script>
""".strip())

    index = SymbolIndex.build(
        [
            result,
        ]
    )

    symbols = index.find_by_name("loadOrders")

    assert len(symbols) == 1

    assert symbols[0].relative_path == "src/views/OrdersView.vue"


def test_nuxt_profile_can_use_vue_sfc_adapter():

    adapter = VueSfcSymbolExtractorAdapter()

    profile = TechnologyProfile(
        languages=["typescript"],
        frameworks=["nuxt"],
    )

    document = SourceDocument(
        relative_path=("pages/orders.vue"),
        content=("<script setup " 'lang="ts">' "</script>"),
    )

    assert adapter.supports(
        profile=profile,
        document=document,
    )
