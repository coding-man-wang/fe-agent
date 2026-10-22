from __future__ import annotations

from pathlib import Path

from agent_core.codebase.graph.extractor import (
    TypeScriptVueImportExtractor,
)
from agent_core.codebase.graph.model import (
    ExternalImportReference,
    GraphFileNode,
    RepositoryDependencyEdge,
    RepositoryDependencyGraph,
    UnresolvedInternalImport,
)
from agent_core.codebase.graph.resolver import ModulePathResolver
from agent_core.codebase.model import RepositoryFileIndex


class RepositoryDependencyGraphBuilder:
    GRAPH_TARGET_SUFFIXES = {
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".vue",
        ".json",
        ".css",
    }

    GRAPH_SOURCE_SUFFIXES = {
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".vue",
    }

    def __init__(
        self,
        *,
        extractor: TypeScriptVueImportExtractor | None = None,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.extractor = extractor or TypeScriptVueImportExtractor()
        self.aliases = aliases

    def build(
        self,
        *,
        file_index: RepositoryFileIndex,
        repository_revision: str | None = None,
    ) -> RepositoryDependencyGraph:
        root = Path(file_index.root_path).resolve(strict=True)

        eligible_records = [
            record
            for record in file_index.files
            if record.is_indexable
            and record.suffix.lower() in self.GRAPH_TARGET_SUFFIXES
        ]

        node_by_path = {
            record.relative_path: GraphFileNode(
                relative_path=record.relative_path,
                category=record.category,
                language=record.language,
            )
            for record in eligible_records
        }

        resolver = ModulePathResolver(
            file_paths=set(node_by_path),
            aliases=self.aliases,
        )

        edges: list[RepositoryDependencyEdge] = []
        external_imports: list[ExternalImportReference] = []
        unresolved: list[UnresolvedInternalImport] = []

        for record in eligible_records:
            if record.suffix.lower() not in self.GRAPH_SOURCE_SUFFIXES:
                continue

            path = root / record.relative_path
            content = path.read_text(encoding="utf-8")

            references = self.extractor.extract(
                relative_path=record.relative_path,
                content=content,
            )

            for reference in references:
                resolution = resolver.resolve(
                    source_path=reference.source_path,
                    specifier=reference.specifier,
                )

                if resolution.kind == "internal":
                    assert resolution.target_path is not None

                    edges.append(
                        RepositoryDependencyEdge(
                            source_path=reference.source_path,
                            target_path=resolution.target_path,
                            import_kind=reference.import_kind,
                            specifier=reference.specifier,
                            line_number=reference.line_number,
                        )
                    )
                    continue

                if resolution.kind == "external":
                    external_imports.append(
                        ExternalImportReference(
                            source_path=reference.source_path,
                            specifier=reference.specifier,
                            import_kind=reference.import_kind,
                            line_number=reference.line_number,
                        )
                    )
                    continue

                unresolved.append(
                    UnresolvedInternalImport(
                        source_path=reference.source_path,
                        specifier=reference.specifier,
                        import_kind=reference.import_kind,
                        line_number=reference.line_number,
                        reason=(resolution.reason or "unresolved_internal_import"),
                    )
                )

        edges.sort(
            key=lambda item: (
                item.source_path,
                item.target_path,
                item.line_number,
                item.import_kind,
            )
        )

        external_imports.sort(
            key=lambda item: (
                item.source_path,
                item.specifier,
                item.line_number,
            )
        )

        unresolved.sort(
            key=lambda item: (
                item.source_path,
                item.specifier,
                item.line_number,
            )
        )

        return RepositoryDependencyGraph(
            root_path=str(root),
            repository_revision=repository_revision,
            nodes=sorted(
                node_by_path.values(),
                key=lambda item: item.relative_path,
            ),
            edges=edges,
            external_imports=external_imports,
            unresolved_internal_imports=unresolved,
        )
