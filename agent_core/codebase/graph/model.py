from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ImportKind = Literal[
    "value",
    "type",
    "side_effect",
    "reexport",
    "dynamic",
]


class ImportReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(min_length=1)
    specifier: str = Field(min_length=1)
    import_kind: ImportKind
    line_number: int = Field(ge=1)


class GraphFileNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str = Field(min_length=1)
    category: str
    language: str | None = None


class RepositoryDependencyEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(min_length=1)
    target_path: str = Field(min_length=1)
    relation: Literal["imports"] = "imports"
    import_kind: ImportKind
    specifier: str = Field(min_length=1)
    line_number: int = Field(ge=1)


class ExternalImportReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(min_length=1)
    specifier: str = Field(min_length=1)
    import_kind: ImportKind
    line_number: int = Field(ge=1)


class UnresolvedInternalImport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(min_length=1)
    specifier: str = Field(min_length=1)
    import_kind: ImportKind
    line_number: int = Field(ge=1)
    reason: str = Field(min_length=1)


class RepositoryDependencyGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    root_path: str
    repository_revision: str | None = None
    nodes: list[GraphFileNode] = Field(default_factory=list)
    edges: list[RepositoryDependencyEdge] = Field(default_factory=list)
    external_imports: list[ExternalImportReference] = Field(default_factory=list)
    unresolved_internal_imports: list[UnresolvedInternalImport] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_graph(self) -> "RepositoryDependencyGraph":
        node_paths = [item.relative_path for item in self.nodes]

        if len(node_paths) != len(set(node_paths)):
            raise ValueError("RepositoryDependencyGraph nodes 存在重复路径")

        node_path_set = set(node_paths)

        edge_keys: set[tuple[str, str, str, int]] = set()

        for edge in self.edges:
            if edge.source_path not in node_path_set:
                raise ValueError(
                    f"Dependency Edge source 不存在于 nodes: {edge.source_path}"
                )

            if edge.target_path not in node_path_set:
                raise ValueError(
                    f"Dependency Edge target 不存在于 nodes: {edge.target_path}"
                )

            key = (
                edge.source_path,
                edge.target_path,
                edge.import_kind,
                edge.line_number,
            )

            if key in edge_keys:
                raise ValueError(
                    "RepositoryDependencyGraph edges 存在重复记录: "
                    f"{edge.source_path} -> {edge.target_path}"
                )

            edge_keys.add(key)

        return self
