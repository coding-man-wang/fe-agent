from pathlib import Path

from agent_core.codebase.file_index import RepositoryFileIndexer
from agent_core.codebase.model import FileRecord
from agent_core.code_intelligence.repository_snapshot import (
    RepositorySymbolSnapshotBuilder,
)


def write(root: Path, relative_path: str, content: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_snapshot_builds_ts_tsx_vue_symbols_and_resolved_graph(tmp_path):
    write(
        tmp_path,
        "src/api/orders.ts",
        """
export interface Order { id: string }
export function listOrders() { return [] }
""",
    )
    write(tmp_path, "src/api/index.ts", 'export * from "./orders"')
    write(
        tmp_path,
        "src/views/Orders.vue",
        """<script setup lang="ts">
import * as api from "@/api"
import type { Order } from "@/api/orders"
function loadOrders(): Order[] { return api.listOrders() }
</script>
<template><main>Orders</main></template>""",
    )
    write(
        tmp_path,
        "src/views/Label.tsx",
        "export const Label = () => <span>Orders</span>",
    )
    index = RepositoryFileIndexer().build(tmp_path)
    snapshot = RepositorySymbolSnapshotBuilder().build(
        file_index=index,
        repository_revision="fixture-revision",
        aliases={"@/": "src/"},
    )
    assert snapshot.repository_revision == "fixture-revision"
    assert snapshot.issues == []
    assert snapshot.symbol_index.find_by_name("Label")
    assert snapshot.symbol_index.find_by_name("Orders")
    edges = snapshot.symbol_graph.all_edges()
    assert any(
        edge.relation == "calls"
        and edge.source_path == "src/views/Orders.vue"
        and edge.target_symbol_id == "src/api/orders.ts::function::listOrders"
        for edge in edges
    )
    assert any(
        edge.relation == "references"
        and edge.target_symbol_id == "src/api/orders.ts::interface::Order"
        for edge in edges
    )
    assert all(
        ".__" not in symbol.relative_path
        for symbol in snapshot.symbol_index.all_symbols()
    )


def test_snapshot_respects_indexability_and_never_reads_escaped_path(tmp_path):
    write(tmp_path, "good.ts", "export function good() {}")
    write(tmp_path, "blocked.ts", "export function blocked() {}")
    write(tmp_path, "ignored.js", "export function ignored() {}")
    index = RepositoryFileIndexer().build(tmp_path)
    next(
        item for item in index.files if item.relative_path == "blocked.ts"
    ).is_indexable = False
    index.files.append(
        FileRecord(
            relative_path="../outside.ts",
            name="outside.ts",
            suffix=".ts",
            size_bytes=1,
            category="source",
            is_indexable=True,
        )
    )
    snapshot = RepositorySymbolSnapshotBuilder().build(
        file_index=index, repository_revision="r1"
    )
    assert {item.name for item in snapshot.symbol_index.all_symbols()} == {"good"}
    assert any("unsafe_repository_path" in issue for issue in snapshot.issues)


def test_snapshot_reports_unreadable_and_newly_oversized_files_but_keeps_healthy_file(
    tmp_path,
):
    write(tmp_path, "good.ts", "export function good() {}")
    write(tmp_path, "missing.ts", "export const missing = 1")
    write(tmp_path, "grown.ts", "export const grown = 1")
    index = RepositoryFileIndexer().build(tmp_path)
    (tmp_path / "missing.ts").unlink()
    write(tmp_path, "grown.ts", "x" * 101)
    snapshot = RepositorySymbolSnapshotBuilder(max_file_bytes=100).build(
        file_index=index,
        repository_revision="r1",
    )
    assert {item.name for item in snapshot.symbol_index.all_symbols()} == {"good"}
    assert any(
        "missing.ts" in issue and "FileNotFoundError" in issue
        for issue in snapshot.issues
    )
    assert any(
        "grown.ts" in issue and "file_too_large" in issue for issue in snapshot.issues
    )


def test_snapshot_rebuild_reads_current_content_and_has_no_stale_cache(tmp_path):
    write(tmp_path, "api.ts", "export function before() {}")
    index = RepositoryFileIndexer().build(tmp_path)
    builder = RepositorySymbolSnapshotBuilder()
    first = builder.build(file_index=index, repository_revision="r1")
    write(tmp_path, "api.ts", "export function after() {}")
    second = builder.build(file_index=index, repository_revision="r2")
    assert first.symbol_index.find_by_name("before")
    assert not second.symbol_index.find_by_name("before")
    assert second.symbol_index.find_by_name("after")
    assert second.repository_revision == "r2"


def test_unresolved_object_member_never_becomes_graph_edge(tmp_path):
    write(tmp_path, "api.ts", "export function listOrders() {}")
    write(
        tmp_path,
        "consumer.ts",
        """
export function load() {
  const store = useOrdersStore()
  return store.listOrders()
}
""",
    )
    snapshot = RepositorySymbolSnapshotBuilder().build(
        file_index=RepositoryFileIndexer().build(tmp_path),
        repository_revision="r1",
    )
    assert snapshot.symbol_graph.all_edges() == []
