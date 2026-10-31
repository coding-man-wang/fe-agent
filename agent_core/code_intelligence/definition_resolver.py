from __future__ import annotations

from agent_core.code_intelligence.model import (
    ImportBinding,
    SymbolDefinitionResolution,
)
from agent_core.code_intelligence.symbol_index import (
    SymbolIndex,
)
from agent_core.codebase.graph.resolver import (
    ModulePathResolver,
)
from agent_core.code_intelligence.module_export import (
    ModuleExportTable,
)
from agent_core.code_intelligence.module_export_resolver import (
    ModuleExportResolver,
)


class SymbolDefinitionResolver:
    """
    Import Binding
        ↓
    Module Resolution
        ↓
    Exported Symbol Lookup
        ↓
    Definition Resolution

    这是 Symbol-level Cross-file
    Understanding 的第一个闭环。
    """

    def __init__(
        self,
        *,
        symbol_index: SymbolIndex,
        file_paths: set[str],
        aliases: dict[str, str] | None = None,
        export_table: ModuleExportTable,
    ) -> None:

        self.symbol_index = symbol_index

        self.module_resolver = ModulePathResolver(
            file_paths=file_paths,
            aliases=aliases,
        )

        self.export_resolver = ModuleExportResolver(
            symbol_index=symbol_index,
            export_table=export_table,
            file_paths=file_paths,
            aliases=aliases,
        )

    def resolve(
        self,
        binding: ImportBinding,
    ) -> SymbolDefinitionResolution:

        module_resolution = self.module_resolver.resolve(
            source_path=(binding.source_path),
            specifier=(binding.module_specifier),
        )

        # ==================================
        # External Dependency
        #
        # vue
        # axios
        # pinia
        #
        # 当前不进入 Repository SymbolIndex。
        # ==================================

        if module_resolution.kind == "external":
            return SymbolDefinitionResolution(
                source_path=(binding.source_path),
                local_name=(binding.local_name),
                imported_name=(binding.imported_name),
                module_specifier=(binding.module_specifier),
                status=("external_module"),
                reason=("bare_module_specifier"),
            )

        # ==================================
        # Repository 内部路径，
        # 但文件不存在。
        # ==================================

        if module_resolution.kind == "unresolved_internal":
            return SymbolDefinitionResolution(
                source_path=(binding.source_path),
                local_name=(binding.local_name),
                imported_name=(binding.imported_name),
                module_specifier=(binding.module_specifier),
                status=("unresolved_module"),
                reason=(module_resolution.reason),
            )

        target_path = module_resolution.target_path

        assert target_path is not None

        # ==================================
        # Side Effect Import
        #
        # 能解析 Module，
        # 但不存在 local symbol definition。
        # ==================================

        if binding.binding_kind == "side_effect":
            return SymbolDefinitionResolution(
                source_path=(binding.source_path),
                local_name=None,
                imported_name=None,
                module_specifier=(binding.module_specifier),
                status="module_only",
                target_path=target_path,
                reason=("side_effect_import"),
            )

        # ==================================
        # Namespace Import
        #
        # import * as api
        #
        # 当前只解析到 Module。
        #
        # api.listOrders
        # 要等 v1.4.4 Reference/Member Graph。
        # ==================================

        if binding.binding_kind == "namespace":
            return SymbolDefinitionResolution(
                source_path=(binding.source_path),
                local_name=(binding.local_name),
                imported_name="*",
                module_specifier=(binding.module_specifier),
                status="module_only",
                target_path=target_path,
                reason=("namespace_import"),
            )

        # ==================================
        # Default Export
        #
        # 当前 CodeSymbol Contract
        # 尚没有 Export Name Table。
        #
        # 不猜。
        # v1.4.3B 正式解决。
        # ==================================

        imported_name = binding.imported_name

        if imported_name is None:
            return SymbolDefinitionResolution(
                source_path=(binding.source_path),
                local_name=(binding.local_name),
                imported_name=None,
                module_specifier=(binding.module_specifier),
                status="unsupported",
                target_path=target_path,
                reason=("missing_imported_name"),
            )

        export_name = (
            "default" if (binding.binding_kind == "default") else binding.imported_name
        )

        if not export_name:
            return SymbolDefinitionResolution(
                source_path=(binding.source_path),
                local_name=(binding.local_name),
                imported_name=(binding.imported_name),
                module_specifier=(binding.module_specifier),
                status="unsupported",
                module_path=target_path,
                reason=("missing_export_name"),
            )

        export_resolution = self.export_resolver.resolve(
            module_path=target_path,
            export_name=export_name,
        )

        return SymbolDefinitionResolution(
            source_path=(binding.source_path),
            local_name=(binding.local_name),
            imported_name=(binding.imported_name),
            module_specifier=(binding.module_specifier),
            status=(export_resolution.status),
            # import specifier 第一跳
            module_path=target_path,
            # 最终真实 definition file
            target_path=(export_resolution.target_path),
            target_symbol_ids=list(export_resolution.target_symbol_ids),
            resolution_chain=list(export_resolution.resolution_chain),
            reason=(export_resolution.reason),
        )

    def resolve_namespace_member(
        self,
        *,
        binding: ImportBinding,
        member_name: str,
    ) -> SymbolDefinitionResolution:
        """
        解析：

            import * as api from "@/api"

            api.listOrders()

        namespace import 本身只解析到 Module。

        member_name 再通过 Module Export Table
        定位最终 Symbol。
        """

        if binding.binding_kind != "namespace":
            return SymbolDefinitionResolution(
                source_path=binding.source_path,
                local_name=binding.local_name,
                imported_name=member_name,
                module_specifier=(binding.module_specifier),
                status="unsupported",
                reason=("binding_is_not_namespace_import"),
            )

        module_resolution = self.module_resolver.resolve(
            source_path=(binding.source_path),
            specifier=(binding.module_specifier),
        )

        if module_resolution.kind == "external":
            return SymbolDefinitionResolution(
                source_path=(binding.source_path),
                local_name=(binding.local_name),
                imported_name=member_name,
                module_specifier=(binding.module_specifier),
                status="external_module",
                reason="bare_module_specifier",
            )

        if (
            module_resolution.kind != "internal"
            or module_resolution.target_path is None
        ):
            return SymbolDefinitionResolution(
                source_path=(binding.source_path),
                local_name=(binding.local_name),
                imported_name=member_name,
                module_specifier=(binding.module_specifier),
                status="unresolved_module",
                reason=(module_resolution.reason),
            )

        module_path = module_resolution.target_path

        export_resolution = self.export_resolver.resolve(
            module_path=module_path,
            export_name=member_name,
        )

        return SymbolDefinitionResolution(
            source_path=(binding.source_path),
            local_name=(binding.local_name),
            imported_name=member_name,
            module_specifier=(binding.module_specifier),
            status=(export_resolution.status),
            module_path=module_path,
            target_path=(export_resolution.target_path),
            target_symbol_ids=list(export_resolution.target_symbol_ids),
            resolution_chain=list(export_resolution.resolution_chain),
            reason=(export_resolution.reason),
        )
