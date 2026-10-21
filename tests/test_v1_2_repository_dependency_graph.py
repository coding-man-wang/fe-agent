from pathlib import Path

from agent_core.codebase.file_index import RepositoryFileIndexer
from agent_core.codebase.graph import (
    ModulePathResolver,
    RepositoryDependencyGraphBuilder,
    RepositoryDependencyGraphQuery,
    TypeScriptVueImportExtractor,
)


def test_vue_import_extractor_keeps_type_and_value_imports():
    content = """<script setup lang=\"ts\">\nimport UserTable from '@/components/UserTable.vue'\nimport { useUsersStore } from '@/stores/users'\nimport type { UserStatus } from '@/types/domain'\n</script>\n"""

    result = TypeScriptVueImportExtractor().extract(
        relative_path="src/views/UsersView.vue",
        content=content,
    )

    assert [
        (
            item.specifier,
            item.import_kind,
        )
        for item in result
    ] == [
        ("@/components/UserTable.vue", "value"),
        ("@/stores/users", "value"),
        ("@/types/domain", "type"),
    ]


def test_module_path_resolver_supports_alias_relative_and_external():
    resolver = ModulePathResolver(
        file_paths={
            "src/views/UsersView.vue",
            "src/stores/users.ts",
            "src/api/users.ts",
            "src/api/requestClient.ts",
        }
    )

    alias_result = resolver.resolve(
        source_path="src/views/UsersView.vue",
        specifier="@/stores/users",
    )

    relative_result = resolver.resolve(
        source_path="src/api/users.ts",
        specifier="./requestClient",
    )

    external_result = resolver.resolve(
        source_path="src/stores/users.ts",
        specifier="pinia",
    )

    assert alias_result.kind == "internal"
    assert alias_result.target_path == "src/stores/users.ts"

    assert relative_result.kind == "internal"
    assert relative_result.target_path == "src/api/requestClient.ts"

    assert external_result.kind == "external"
    assert external_result.target_path is None


def test_builder_and_query_build_real_file_dependency_chain(tmp_path: Path):
    src = tmp_path / "src"

    (src / "views").mkdir(parents=True)
    (src / "stores").mkdir(parents=True)
    (src / "api").mkdir(parents=True)
    (src / "types").mkdir(parents=True)

    (src / "views" / "UsersView.vue").write_text(
        """<script setup lang=\"ts\">\nimport { useUsersStore } from '@/stores/users'\nimport type { UserStatus } from '@/types/domain'\n</script>\n""",
        encoding="utf-8",
    )

    (src / "stores" / "users.ts").write_text(
        """import { defineStore } from 'pinia'\nimport { listUsers } from '@/api/users'\nimport type { UserQuery } from '@/types/domain'\nexport const useUsersStore = defineStore('users', () => ({ listUsers }))\n""",
        encoding="utf-8",
    )

    (src / "api" / "users.ts").write_text(
        """import { request } from './requestClient'\nimport type { UserQuery } from '@/types/domain'\nexport function listUsers(query: UserQuery) { return request(query) }\n""",
        encoding="utf-8",
    )

    (src / "api" / "requestClient.ts").write_text(
        "export function request(value: unknown) { return value }\n",
        encoding="utf-8",
    )

    (src / "types" / "domain.ts").write_text(
        "export type UserStatus = 'active' | 'disabled'\nexport interface UserQuery { page: number }\n",
        encoding="utf-8",
    )

    file_index = RepositoryFileIndexer().build(tmp_path)

    graph = RepositoryDependencyGraphBuilder().build(
        file_index=file_index,
        repository_revision="a" * 40,
    )

    edge_pairs = {
        (
            edge.source_path,
            edge.target_path,
        )
        for edge in graph.edges
    }

    assert (
        "src/views/UsersView.vue",
        "src/stores/users.ts",
    ) in edge_pairs

    assert (
        "src/stores/users.ts",
        "src/api/users.ts",
    ) in edge_pairs

    assert (
        "src/api/users.ts",
        "src/api/requestClient.ts",
    ) in edge_pairs

    assert (
        "src/api/users.ts",
        "src/types/domain.ts",
    ) in edge_pairs

    assert any(
        item.source_path == "src/stores/users.ts"
        and item.specifier == "pinia"
        for item in graph.external_imports
    )

    query = RepositoryDependencyGraphQuery(graph)

    dependencies = query.dependencies(
        "src/views/UsersView.vue",
        max_depth=3,
    )

    dependency_paths = {
        item.relative_path
        for item in dependencies
    }

    assert "src/stores/users.ts" in dependency_paths
    assert "src/api/users.ts" in dependency_paths
    assert "src/types/domain.ts" in dependency_paths

    dependents = query.dependents(
        "src/api/users.ts",
        max_depth=2,
    )

    dependent_paths = {
        item.relative_path
        for item in dependents
    }

    assert "src/stores/users.ts" in dependent_paths
    assert "src/views/UsersView.vue" in dependent_paths


def test_builder_records_unresolved_internal_import(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()

    (src / "main.ts").write_text(
        "import { missing } from '@/missing/module'\n",
        encoding="utf-8",
    )

    file_index = RepositoryFileIndexer().build(tmp_path)

    graph = RepositoryDependencyGraphBuilder().build(
        file_index=file_index,
    )

    assert len(graph.unresolved_internal_imports) == 1
    assert (
        graph.unresolved_internal_imports[0].specifier
        == "@/missing/module"
    )
