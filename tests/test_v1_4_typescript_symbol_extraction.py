from agent_core.code_intelligence.adapters.typescript import (
    TypeScriptSymbolExtractorAdapter,
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
    return TechnologyProfile(languages=["typescript"])


def extract(
    content: str,
    *,
    relative_path: str = ("src/api/orders.ts"),
):
    adapter = TypeScriptSymbolExtractorAdapter()

    profile = make_profile()

    document = SourceDocument(
        relative_path=relative_path,
        content=content,
        language="typescript",
    )

    assert adapter.supports(
        profile=profile,
        document=document,
    )

    return adapter.extract_symbols(
        profile=profile,
        document=document,
    )


def test_extracts_basic_typescript_symbols():

    result = extract("""
export interface OrderQuery {
  status?: string
}

export type OrderStatus =
  "pending" | "paid"

const DEFAULT_LIMIT = 20

export function listOrders() {
  return []
}
""".strip())

    by_name = {symbol.name: symbol for symbol in result.symbols}

    assert by_name["OrderQuery"].kind == "interface"

    assert by_name["OrderQuery"].exported is True

    assert by_name["OrderStatus"].kind == "type"

    assert by_name["OrderStatus"].exported is True

    assert by_name["DEFAULT_LIMIT"].kind == "variable"

    assert by_name["DEFAULT_LIMIT"].exported is False

    assert by_name["listOrders"].kind == "function"

    assert by_name["listOrders"].exported is True


def test_arrow_function_is_engineering_function():

    result = extract("""
export const loadOrders = async () => {
  return []
}

const normalizeStatus = (value: string) => {
  return value.toLowerCase()
}
""".strip())

    by_name = {symbol.name: symbol for symbol in result.symbols}

    assert by_name["loadOrders"].kind == "function"

    assert by_name["loadOrders"].exported is True

    assert by_name["normalizeStatus"].kind == "function"

    assert by_name["normalizeStatus"].exported is False


def test_class_method_has_parent_symbol():

    result = extract(
        """
export class OrderService {
  listOrders() {
    return []
  }

  cancelOrder() {
    return true
  }
}
""".strip(),
        relative_path=("src/services/" "orderService.ts"),
    )

    by_name = {symbol.name: symbol for symbol in result.symbols}

    class_symbol = by_name["OrderService"]

    list_method = by_name["listOrders"]

    cancel_method = by_name["cancelOrder"]

    assert class_symbol.kind == "class"

    assert list_method.kind == "method"

    assert list_method.parent_symbol_id == class_symbol.symbol_id

    assert cancel_method.parent_symbol_id == class_symbol.symbol_id


def test_nested_function_has_parent_symbol():

    result = extract("""
export function loadOrders() {
  function normalize() {
    return true
  }

  return normalize()
}
""".strip())

    by_name = {symbol.name: symbol for symbol in result.symbols}

    outer = by_name["loadOrders"]

    inner = by_name["normalize"]

    assert inner.parent_symbol_id == outer.symbol_id


def test_tsx_uses_tsx_grammar():

    result = extract(
        """
export const OrdersPage = () => {
  return (
    <div>
      Orders
    </div>
  )
}
""".strip(),
        relative_path=("src/pages/" "OrdersPage.tsx"),
    )

    symbols = {symbol.name: symbol for symbol in result.symbols}
    print(symbols)
    assert "OrdersPage" in symbols

    assert symbols["OrdersPage"].kind == "function"

    assert result.issues == []


def test_syntax_error_does_not_destroy_recoverable_symbols():

    result = extract("""
export function validFunction() {
  return 1
}

export function broken( {
""".strip())

    names = {symbol.name for symbol in result.symbols}

    assert "validFunction" in names

    assert len(result.issues) >= 1


def test_symbol_ids_are_unique_for_overloads():

    result = extract("""
export function parse(value: string): string
export function parse(value: number): number
export function parse(
  value: string | number
): string | number {
  return value
}
""".strip())

    parse_symbols = [symbol for symbol in result.symbols if symbol.name == "parse"]

    symbol_ids = {symbol.symbol_id for symbol in parse_symbols}

    assert len(symbol_ids) == len(parse_symbols)


def test_symbol_index_supports_name_and_file_lookup():

    orders = extract(
        """
export function load() {
  return []
}
""".strip(),
        relative_path=("src/api/orders.ts"),
    )

    users = extract(
        """
export function load() {
  return []
}
""".strip(),
        relative_path=("src/api/users.ts"),
    )

    index = SymbolIndex.build(
        [
            orders,
            users,
        ]
    )

    load_symbols = index.find_by_name("load")

    assert len(load_symbols) == 2

    assert {item.relative_path for item in load_symbols} == {
        "src/api/orders.ts",
        "src/api/users.ts",
    }

    orders_symbols = index.find_by_file("src/api/orders.ts")

    assert len(orders_symbols) == 1

    assert orders_symbols[0].name == "load"


def test_registry_resolves_typescript_adapter():

    registry = SymbolExtractorRegistry()

    adapter = TypeScriptSymbolExtractorAdapter()

    registry.register(adapter)

    profile = make_profile()

    document = SourceDocument(
        relative_path=("src/api/orders.ts"),
        content=("export function " "listOrders() {}"),
        language="typescript",
    )

    resolved = registry.resolve(
        profile=profile,
        document=document,
    )

    assert resolved.adapter_id == ("tree-sitter-typescript")


def test_typescript_adapter_rejects_javascript_file():

    adapter = TypeScriptSymbolExtractorAdapter()

    profile = make_profile()

    document = SourceDocument(
        relative_path=("src/api/orders.js"),
        content="",
        language="javascript",
    )

    assert (
        adapter.supports(
            profile=profile,
            document=document,
        )
        is False
    )
