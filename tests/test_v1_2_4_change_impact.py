from agent_core.codebase.graph.model import (
    GraphFileNode,
    RepositoryDependencyEdge,
    RepositoryDependencyGraph,
)
from agent_core.codebase.impact import (
    RepositoryChangeImpactAnalyzer,
)
from agent_core.codebase.model import (
    FileRecord,
    RepositoryFileIndex,
)


def _record(
    *,
    path: str,
    category: str,
) -> FileRecord:

    name = path.rsplit("/", 1)[-1]

    suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""

    return FileRecord(
        relative_path=path,
        name=name,
        suffix=suffix,
        size_bytes=100,
        category=category,
        language="TypeScript",
        is_indexable=True,
    )


def _node(
    *,
    path: str,
    category: str,
) -> GraphFileNode:

    return GraphFileNode(
        relative_path=path,
        category=category,
        language="TypeScript",
    )


def _edge(
    *,
    source: str,
    target: str,
) -> RepositoryDependencyEdge:

    return RepositoryDependencyEdge(
        source_path=source,
        target_path=target,
        import_kind="value",
        specifier=target,
        line_number=1,
    )


def _index() -> RepositoryFileIndex:

    files = [
        _record(
            path="src/stores/users.ts",
            category="source",
        ),
        _record(
            path="src/views/UsersView.vue",
            category="source",
        ),
        _record(
            path="src/router/index.ts",
            category="source",
        ),
        _record(
            path=("tests/unit/" "usersStore.test.ts"),
            category="test",
        ),
        _record(
            path=("tests/e2e/users.spec.ts"),
            category="test",
        ),
    ]

    return RepositoryFileIndex(
        root_path="/repo",
        total_files=len(files),
        indexable_files=len(files),
        files=files,
        category_counts={
            "source": 3,
            "test": 2,
        },
    )


def _graph() -> RepositoryDependencyGraph:

    nodes = [
        _node(
            path="src/stores/users.ts",
            category="source",
        ),
        _node(
            path="src/views/UsersView.vue",
            category="source",
        ),
        _node(
            path="src/router/index.ts",
            category="source",
        ),
        _node(
            path=("tests/unit/" "usersStore.test.ts"),
            category="test",
        ),
        _node(
            path=("tests/e2e/users.spec.ts"),
            category="test",
        ),
    ]

    edges = [
        # UsersView 直接依赖 users store。
        _edge(
            source=("src/views/UsersView.vue"),
            target=("src/stores/users.ts"),
        ),
        # router 间接受到 users store 影响。
        _edge(
            source=("src/router/index.ts"),
            target=("src/views/UsersView.vue"),
        ),
        # unit test 直接 import store。
        _edge(
            source=("tests/unit/" "usersStore.test.ts"),
            target=("src/stores/users.ts"),
        ),
        # E2E 不直接 import users.ts，
        # 因此 Graph 不会自动找到它。
    ]

    return RepositoryDependencyGraph(
        root_path="/repo",
        repository_revision="commit-a",
        nodes=nodes,
        edges=edges,
        external_imports=[],
        unresolved_internal_imports=[],
    )


def test_change_impact_finds_direct_and_indirect_dependents():
    analyzer = RepositoryChangeImpactAnalyzer(max_depth=2)

    report = analyzer.analyze(
        changed_files=["src/stores/users.ts"],
        graph=_graph(),
        index=_index(),
        repository_revision=("commit-a"),
        known_test_files=["tests/e2e/users.spec.ts"],
    )

    assert report.changed_files == ["src/stores/users.ts"]

    production = {item.relative_path: item for item in report.affected_production_files}

    assert "src/views/UsersView.vue" in production

    assert production["src/views/UsersView.vue"].depth == 1

    assert "src/router/index.ts" in production

    assert production["src/router/index.ts"].depth == 2

    tests = {item.relative_path for item in report.affected_test_files}

    assert "tests/unit/usersStore.test.ts" in tests

    # Graph 找到 unit，
    # Change Surface 补充 e2e。
    assert report.suggested_test_files == [
        "tests/unit/usersStore.test.ts",
        "tests/e2e/users.spec.ts",
    ]

    assert report.impact_level == "medium"


def test_change_impact_does_not_include_changed_file_itself():
    analyzer = RepositoryChangeImpactAnalyzer(max_depth=2)

    report = analyzer.analyze(
        changed_files=[
            "src/stores/users.ts",
            "src/views/UsersView.vue",
        ],
        graph=_graph(),
        index=_index(),
        repository_revision=("commit-a"),
    )

    impacted_paths = {
        item.relative_path
        for item in [
            *report.affected_production_files,
            *report.affected_test_files,
        ]
    }

    assert "src/stores/users.ts" not in impacted_paths

    assert "src/views/UsersView.vue" not in impacted_paths


def test_change_impact_is_unknown_when_changed_file_is_not_in_graph():
    analyzer = RepositoryChangeImpactAnalyzer()

    report = analyzer.analyze(
        changed_files=["vite.config.ts"],
        graph=_graph(),
        index=_index(),
        repository_revision=("commit-a"),
    )

    assert report.impact_level == "unknown"

    assert report.unresolved_changed_files == ["vite.config.ts"]
