from pathlib import Path

from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.model import (
    CodeChunk,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)
from agent_core.code_intelligence.repository_snapshot import (
    RepositorySymbolSnapshotBuilder,
)
from agent_core.context.symbol_window import (
    SymbolWindowBuilder,
)


def candidate(
    *,
    symbol_id: str,
    name: str,
    kind: str,
    path: str,
    start: int,
    end: int,
) -> SymbolCandidate:

    return SymbolCandidate(
        symbol_id=symbol_id,
        name=name,
        kind=kind,
        relative_path=path,
        start_line=start,
        end_line=end,
        score=0.9,
        source="evidence",
    )


def test_function_window_keeps_imports_and_symbol():

    lines = [
        "import { request } " "from './requestClient'",
        "import type { Order } " "from './domain'",
        *[
            f"const filler_{line} = {line}"
            for line in range(
                3,
                61,
            )
        ],
        "export function " "target(): Order {",
        "  return " "request<Order>()",
        "}",
        *[
            f"const filler_{line} = {line}"
            for line in range(
                64,
                121,
            )
        ],
    ]

    content = "\n".join(lines)

    symbol = candidate(
        symbol_id=("src/api/demo.ts" "::function::target"),
        name="target",
        kind="function",
        path="src/api/demo.ts",
        start=61,
        end=63,
    )

    result = SymbolWindowBuilder(context_lines=1).build(
        relative_path=("src/api/demo.ts"),
        content=content,
        candidates=[symbol],
        semantic_chunk=None,
        max_chars=800,
    )

    assert result.usable is True

    assert result.worthwhile is True

    assert symbol.symbol_id in result.covered_symbol_ids

    assert "import { request }" in result.content

    assert "export function target()" in result.content

    assert "filler_100" not in result.content

    assert result.reduction_ratio > 0.50


def test_optional_context_cannot_evict_mandatory_symbol():

    content = "\n".join(
        [
            "const a = 1",
            "const b = 2",
            "const c = 3",
            "export function target() {",
            "  return 1",
            "}",
            "const d = 4",
            "const e = 5",
        ]
    )

    exact = "\n".join(
        [
            "export function target() {",
            "  return 1",
            "}",
        ]
    )

    symbol = candidate(
        symbol_id=("src/demo.ts" "::function::target"),
        name="target",
        kind="function",
        path="src/demo.ts",
        start=4,
        end=6,
    )

    result = SymbolWindowBuilder(context_lines=3).build(
        relative_path=("src/demo.ts"),
        content=content,
        candidates=[symbol],
        semantic_chunk=None,
        max_chars=len(exact),
    )

    assert result.usable is True

    assert result.content == exact

    assert result.segments[0].start_line == 4

    assert result.segments[0].end_line == 6


def test_window_fails_closed_when_symbol_exceeds_budget():

    content = "\n".join(
        [
            "export function huge() {",
            ("  const a = '" + ("x" * 200) + "'"),
            ("  const b = '" + ("y" * 200) + "'"),
            "  return a + b",
            "}",
        ]
    )

    symbol_id = "src/huge.ts" "::function::huge"

    result = SymbolWindowBuilder().build(
        relative_path=("src/huge.ts"),
        content=content,
        candidates=[
            candidate(
                symbol_id=(symbol_id),
                name="huge",
                kind="function",
                path="src/huge.ts",
                start=1,
                end=5,
            )
        ],
        semantic_chunk=None,
        max_chars=80,
    )

    assert result.usable is False

    assert symbol_id in result.missing_symbol_ids

    assert "mandatory_symbol_evidence_exceeds_budget" in result.issues


def test_container_uses_semantic_localization():

    lines = [
        f"const line_{line} = {line}"
        for line in range(
            1,
            101,
        )
    ]

    content = "\n".join(lines)

    component_id = "src/views/OrdersView.vue" "::component::OrdersView"

    semantic_chunk = CodeChunk(
        chunk_id=("src/views/" "OrdersView.vue:20-30"),
        relative_path=("src/views/OrdersView.vue"),
        start_line=20,
        end_line=30,
        content="\n".join(lines[19:30]),
        language="Vue",
        category="source",
    )

    result = SymbolWindowBuilder().build(
        relative_path=("src/views/" "OrdersView.vue"),
        content=content,
        candidates=[
            candidate(
                symbol_id=(component_id),
                name="OrdersView",
                kind="component",
                path=("src/views/" "OrdersView.vue"),
                start=1,
                end=100,
            )
        ],
        semantic_chunk=(semantic_chunk),
        max_chars=500,
    )

    assert result.usable is True

    assert component_id in result.localized_symbol_ids

    assert component_id not in result.covered_symbol_ids

    assert "line_20" in result.content

    assert "line_90" not in result.content


def test_vue_window_uses_real_repository_lines(
    tmp_path: Path,
):

    vue_path = tmp_path / "src" / "views" / "OrdersView.vue"

    vue_path.parent.mkdir(parents=True)

    content = """<template>
  <div>{{ orders.length }}</div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const orders = ref<string[]>([])
</script>
"""

    vue_path.write_text(
        content,
        encoding="utf-8",
    )

    index = RepositoryFileIndexer().build(tmp_path)

    snapshot = RepositorySymbolSnapshotBuilder().build(
        file_index=index,
        repository_revision=("vue-window-test"),
    )

    orders = next(
        symbol
        for symbol in snapshot.symbol_index.find_by_file("src/views/OrdersView.vue")
        if (symbol.name == "orders" and symbol.kind == "variable")
    )

    assert orders.relative_path == "src/views/OrdersView.vue"

    assert orders.start_line == 8

    semantic_chunk = CodeChunk(
        chunk_id=("src/views/" "OrdersView.vue:6-9"),
        relative_path=("src/views/OrdersView.vue"),
        start_line=6,
        end_line=9,
        content="\n".join(content.splitlines()[5:9]),
        language="Vue",
        category="source",
    )

    symbol_candidate = SymbolCandidate(
        **orders.model_dump(
            include={
                "symbol_id",
                "name",
                "kind",
                "relative_path",
                "start_line",
                "end_line",
            }
        ),
        score=0.9,
        source="evidence",
    )

    result = SymbolWindowBuilder().build(
        relative_path=("src/views/" "OrdersView.vue"),
        content=content,
        candidates=[symbol_candidate],
        semantic_chunk=(semantic_chunk),
        max_chars=500,
    )

    assert result.usable is True

    assert orders.symbol_id in result.covered_symbol_ids

    assert "const orders = " "ref<string[]>([])" in result.content
