from __future__ import annotations

from dataclasses import dataclass

from agent_core.code_intelligence.module_export import (
    ModuleExportTable,
)
from agent_core.code_intelligence.symbol_index import (
    SymbolIndex,
)
from agent_core.codebase.graph.resolver import (
    ModulePathResolver,
)


@dataclass(frozen=True)
class ModuleExportResolution:
    status: str

    target_symbol_ids: tuple[str, ...] = ()

    target_path: str | None = None

    resolution_chain: tuple[str, ...] = ()

    reason: str | None = None


class ModuleExportResolver:
    """
    Module Export Name
        ↓
    local export /
    named re-export /
    star re-export
        ↓
    final CodeSymbol

    支持：
        recursive barrel
        cycle protection
        ambiguity detection
    """

    def __init__(
        self,
        *,
        symbol_index: SymbolIndex,
        export_table: ModuleExportTable,
        file_paths: set[str],
        aliases: dict[str, str] | None = None,
    ) -> None:

        self.symbol_index = symbol_index

        self.export_table = export_table

        self.module_resolver = ModulePathResolver(
            file_paths=file_paths,
            aliases=aliases,
        )

    def resolve(
        self,
        *,
        module_path: str,
        export_name: str,
    ) -> ModuleExportResolution:

        return self._resolve(
            module_path=module_path,
            export_name=export_name,
            visited=frozenset(),
        )

    def _resolve(
        self,
        *,
        module_path: str,
        export_name: str,
        visited: frozenset[tuple[str, str]],
    ) -> ModuleExportResolution:

        key = (
            module_path,
            export_name,
        )

        chain_item = f"{module_path}" f"::{export_name}"

        if key in visited:

            return ModuleExportResolution(
                status="cyclic_export",
                resolution_chain=(chain_item,),
                reason=("circular_export_chain"),
            )

        next_visited = visited | {key}

        # ==================================
        # Explicit Export 优先
        #
        # export { foo }
        #
        # 优先于：
        #
        # export * from "./x"
        # ==================================

        explicit = self.export_table.find_explicit(
            relative_path=(module_path),
            exported_name=(export_name),
        )

        if explicit:

            results = [
                self._resolve_binding(
                    binding=item,
                    visited=next_visited,
                )
                for item in explicit
            ]

            return self._combine(
                results=results,
                chain_item=chain_item,
            )

        # ==================================
        # export * 不传播 default
        # ==================================

        if export_name == "default":

            return ModuleExportResolution(
                status="export_not_found",
                resolution_chain=(chain_item,),
            )

        # ==================================
        # Star Re-export
        # ==================================

        star_exports = self.export_table.find_star_exports(module_path)

        results: list[ModuleExportResolution] = []

        for binding in star_exports:

            module_resolution = self.module_resolver.resolve(
                source_path=(module_path),
                specifier=(binding.module_specifier or ""),
            )

            if module_resolution.kind != "internal" or (
                module_resolution.target_path is None
            ):
                continue

            results.append(
                self._resolve(
                    module_path=(module_resolution.target_path),
                    export_name=(export_name),
                    visited=next_visited,
                )
            )

        if not results:

            return ModuleExportResolution(
                status="export_not_found",
                resolution_chain=(chain_item,),
            )

        return self._combine(
            results=results,
            chain_item=chain_item,
        )

    def _resolve_binding(
        self,
        *,
        binding,
        visited,
    ) -> ModuleExportResolution:

        # ==================================
        # Local Export
        # ==================================

        if binding.binding_kind == "local":

            if not binding.local_name:

                return ModuleExportResolution(
                    status="unsupported",
                    reason=("anonymous_local_export"),
                )

            symbols = self.symbol_index.find_by_file_and_name(
                relative_path=(binding.source_path),
                name=(binding.local_name),
            )

            if not symbols:

                return ModuleExportResolution(
                    status=("symbol_not_found"),
                    target_path=(binding.source_path),
                    reason=("local_export_symbol_" "not_found"),
                )

            return ModuleExportResolution(
                status="resolved",
                target_symbol_ids=tuple(item.symbol_id for item in symbols),
                target_path=(binding.source_path),
            )

        # ==================================
        # Named Re-export
        # ==================================

        if binding.binding_kind == "reexport":

            module_resolution = self.module_resolver.resolve(
                source_path=(binding.source_path),
                specifier=(binding.module_specifier or ""),
            )

            if module_resolution.kind == "external":

                return ModuleExportResolution(
                    status=("external_module"),
                    reason=("external_reexport"),
                )

            if module_resolution.kind != "internal" or (
                module_resolution.target_path is None
            ):

                return ModuleExportResolution(
                    status=("unresolved_module"),
                    reason=(module_resolution.reason),
                )

            return self._resolve(
                module_path=(module_resolution.target_path),
                export_name=(binding.imported_name or ""),
                visited=visited,
            )

        return ModuleExportResolution(
            status="unsupported",
        )

    def _combine(
        self,
        *,
        results: list[ModuleExportResolution],
        chain_item: str,
    ) -> ModuleExportResolution:

        resolved = [item for item in results if item.status == "resolved"]

        # 同一 overload set / 同一最终 Symbol
        # 去重。
        unique_targets = {tuple(sorted(item.target_symbol_ids)) for item in resolved}

        if len(unique_targets) > 1:

            return ModuleExportResolution(
                status="ambiguous_export",
                resolution_chain=(chain_item,),
                reason=("multiple_star_exports_" "resolve_same_name"),
            )

        if resolved:

            selected = resolved[0]

            return ModuleExportResolution(
                status="resolved",
                target_symbol_ids=(selected.target_symbol_ids),
                target_path=(selected.target_path),
                resolution_chain=((chain_item,) + selected.resolution_chain),
            )

        cyclic = next(
            (item for item in results if item.status == "cyclic_export"),
            None,
        )

        if cyclic is not None:

            return ModuleExportResolution(
                status="cyclic_export",
                resolution_chain=((chain_item,) + cyclic.resolution_chain),
                reason=(cyclic.reason),
            )

        # 没有 resolved，则保留第一个
        # 有信息的失败结果。
        selected = results[0]

        return ModuleExportResolution(
            status=selected.status,
            target_symbol_ids=(selected.target_symbol_ids),
            target_path=(selected.target_path),
            resolution_chain=((chain_item,) + selected.resolution_chain),
            reason=selected.reason,
        )
