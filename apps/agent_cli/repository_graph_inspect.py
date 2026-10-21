from __future__ import annotations

import argparse
from pathlib import Path

from agent_core.codebase.file_index import RepositoryFileIndexer
from agent_core.codebase.graph import (
    RepositoryDependencyGraphBuilder,
    RepositoryDependencyGraphQuery,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and inspect the repository dependency graph.",
    )

    parser.add_argument(
        "repository_root",
        help="Target repository root path",
    )

    parser.add_argument(
        "--file",
        dest="relative_path",
        default="src/views/UsersView.vue",
        help="Repository-relative file to inspect",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Dependency traversal depth",
    )

    parser.add_argument(
        "--revision",
        default=None,
        help="Optional immutable repository revision",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    repository_root = Path(args.repository_root).resolve(strict=True)

    file_index = RepositoryFileIndexer().build(repository_root)

    graph = RepositoryDependencyGraphBuilder().build(
        file_index=file_index,
        repository_revision=args.revision,
    )

    query = RepositoryDependencyGraphQuery(graph)

    print("=== Repository Dependency Graph ===")
    print(f"root={graph.root_path}")
    print(f"revision={graph.repository_revision}")
    print(f"nodes={len(graph.nodes)}")
    print(f"edges={len(graph.edges)}")
    print(f"external_imports={len(graph.external_imports)}")
    print(
        "unresolved_internal_imports="
        f"{len(graph.unresolved_internal_imports)}"
    )

    print()
    print(f"=== Dependencies: {args.relative_path} ===")

    for hit in query.dependencies(
        args.relative_path,
        max_depth=args.depth,
    ):
        print(
            f"depth={hit.depth} "
            f"via={hit.via_path} "
            f"target={hit.relative_path}"
        )

    print()
    print(f"=== Dependents: {args.relative_path} ===")

    for hit in query.dependents(
        args.relative_path,
        max_depth=args.depth,
    ):
        print(
            f"depth={hit.depth} "
            f"via={hit.via_path} "
            f"source={hit.relative_path}"
        )

    if graph.unresolved_internal_imports:
        print()
        print("=== Unresolved Internal Imports ===")

        for item in graph.unresolved_internal_imports:
            print(
                f"{item.source_path}:{item.line_number} "
                f"{item.specifier} -> {item.reason}"
            )

    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            graph.model_dump_json(indent=2),
            encoding="utf-8",
        )
        print()
        print(f"graph_json={output}")


if __name__ == "__main__":
    main()
