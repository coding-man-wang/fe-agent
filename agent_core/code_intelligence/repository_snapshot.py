"""从当前可索引文件构建只读符号快照；不决定文件修改权限。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from agent_core.codebase.model import RepositoryFileIndex
from agent_core.code_intelligence.adapters.typescript import (
    TypeScriptSymbolExtractorAdapter,
)
from agent_core.code_intelligence.adapters.typescript_exports import (
    TypeScriptModuleExportExtractor,
)
from agent_core.code_intelligence.adapters.typescript_imports import (
    TypeScriptImportBindingExtractorAdapter,
)
from agent_core.code_intelligence.adapters.typescript_references import (
    TypeScriptSymbolReferenceExtractor,
)
from agent_core.code_intelligence.adapters.vue import VueSfcSymbolExtractorAdapter
from agent_core.code_intelligence.definition_resolver import SymbolDefinitionResolver
from agent_core.code_intelligence.model import SourceDocument, TechnologyProfile
from agent_core.code_intelligence.module_export import ModuleExportTable
from agent_core.code_intelligence.reference_resolver import SymbolReferenceResolver
from agent_core.code_intelligence.symbol_graph import (
    SymbolGraph,
    SymbolReferenceGraphBuilder,
)
from agent_core.code_intelligence.symbol_index import SymbolIndex


@dataclass(frozen=True)
class RepositorySymbolSnapshot:
    repository_revision: str
    symbol_index: SymbolIndex
    symbol_graph: SymbolGraph
    issues: list[str] = field(default_factory=list)


class RepositorySymbolSnapshotBuilder:
    """现读现建，无持久缓存；单文件失败只记录问题，不阻断文件检索。"""

    def __init__(self, *, max_file_bytes: int = 1_000_000) -> None:
        if max_file_bytes < 1:
            raise ValueError("max_file_bytes 必须大于零")
        self.max_file_bytes = max_file_bytes

    def build(
        self,
        *,
        file_index: RepositoryFileIndex,
        repository_revision: str,
        aliases: dict[str, str] | None = None,
    ) -> RepositorySymbolSnapshot:
        root = Path(file_index.root_path).resolve()
        # 技术适配集中于此边界；SymbolIndex 和检索层均不感知具体语法。
        profile = TechnologyProfile(languages=["typescript"], frameworks=["vue"])
        ts_symbols = TypeScriptSymbolExtractorAdapter()
        ts_imports = TypeScriptImportBindingExtractorAdapter()
        ts_exports = TypeScriptModuleExportExtractor()
        ts_references = TypeScriptSymbolReferenceExtractor()
        vue = VueSfcSymbolExtractorAdapter()
        symbol_results = []
        export_results = []
        imports = []
        occurrences = []
        issues: list[str] = []
        parsed_paths: set[str] = set()

        for record in sorted(file_index.files, key=lambda item: item.relative_path):
            relative_path = record.relative_path
            suffix = PurePosixPath(relative_path).suffix.lower()
            if not record.is_indexable or suffix not in {".ts", ".tsx", ".vue"}:
                continue
            if relative_path in parsed_paths:
                continue
            try:
                path = self._checked_path(root, relative_path)
                with path.open("rb") as stream:
                    raw = stream.read(self.max_file_bytes + 1)
                if len(raw) > self.max_file_bytes:
                    raise ValueError("file_too_large")
                document = SourceDocument(
                    relative_path=relative_path,
                    content=raw.decode("utf-8-sig"),
                    language="tsx" if suffix == ".tsx" else "typescript",
                )
                symbol_adapter = vue if suffix == ".vue" else ts_symbols
                import_adapter = vue if suffix == ".vue" else ts_imports
                export_adapter = vue if suffix == ".vue" else ts_exports
                reference_adapter = vue if suffix == ".vue" else ts_references
                symbols = symbol_adapter.extract_symbols(
                    profile=profile, document=document
                )
                bindings = import_adapter.extract_import_bindings(
                    profile=profile, document=document
                )
                exports = export_adapter.extract_module_exports(
                    profile=profile, document=document
                )
                references = reference_adapter.extract_references(
                    profile=profile, document=document
                )
                for result in (symbols, bindings, exports, references):
                    issues.extend(
                        f"{relative_path}: {issue.message}" for issue in result.issues
                    )
                symbol_results.append(symbols)
                export_results.append(exports)
                imports.extend(bindings.bindings)
                occurrences.extend(references.references)
                parsed_paths.add(relative_path)
            except Exception as error:
                # 可选符号通道的错误不得影响原文件通道；保留可诊断原因。
                issues.append(f"{relative_path}: {type(error).__name__}: {error}")

        symbol_index = SymbolIndex.build(symbol_results)
        definitions = SymbolDefinitionResolver(
            symbol_index=symbol_index,
            export_table=ModuleExportTable.build(export_results),
            file_paths=parsed_paths,
            aliases=aliases,
        )
        resolver = SymbolReferenceResolver(
            symbol_index=symbol_index,
            definition_resolver=definitions,
            import_bindings=imports,
        )
        resolutions = []
        for occurrence in occurrences:
            try:
                resolutions.append(resolver.resolve(occurrence))
            except Exception as error:
                issues.append(
                    f"{occurrence.source_path}:{occurrence.line_number}: "
                    f"{type(error).__name__}: {error}"
                )
        return RepositorySymbolSnapshot(
            repository_revision=repository_revision,
            symbol_index=symbol_index,
            symbol_graph=SymbolReferenceGraphBuilder().build(resolutions),
            issues=issues,
        )

    @staticmethod
    def _checked_path(root: Path, relative_path: str) -> Path:
        logical = PurePosixPath(relative_path)
        if (
            logical.is_absolute()
            or ".." in logical.parts
            or "\\" in relative_path
            or ":" in relative_path
        ):
            raise ValueError("unsafe_repository_path")
        candidate = root.joinpath(*logical.parts)
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root) or candidate.is_symlink():
            raise ValueError("unsafe_repository_path")
        return resolved
