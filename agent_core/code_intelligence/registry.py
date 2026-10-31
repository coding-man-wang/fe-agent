from __future__ import annotations

from agent_core.code_intelligence.model import (
    SourceDocument,
    TechnologyProfile,
)
from agent_core.code_intelligence.protocols import (
    SymbolExtractorAdapter,
)


class CodeIntelligenceAdapterError(ValueError):
    pass


class DuplicateAdapterError(CodeIntelligenceAdapterError):
    pass


class AdapterNotFoundError(CodeIntelligenceAdapterError):
    pass


class AmbiguousAdapterError(CodeIntelligenceAdapterError):
    pass


class SymbolExtractorRegistry:
    """
    Technology Adapter Registry。

    Core Module 不应该：

        if framework == "Vue3":
        elif framework == "React":

    Core 只负责：

        registry.resolve(...)

    Framework-specific decision
    必须终止在 Adapter Boundary。
    """

    def __init__(
        self,
    ) -> None:

        self._adapters: dict[
            str,
            SymbolExtractorAdapter,
        ] = {}

    def register(
        self,
        adapter: SymbolExtractorAdapter,
    ) -> None:

        adapter_id = adapter.adapter_id

        if not adapter_id:
            raise ValueError("adapter_id 不能为空")

        if adapter_id in self._adapters:
            raise DuplicateAdapterError(
                "重复注册 Code Intelligence " f"Adapter: {adapter_id}"
            )

        self._adapters[adapter_id] = adapter

    def resolve(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> SymbolExtractorAdapter:

        candidates = [
            adapter
            for adapter in self._adapters.values()
            if adapter.supports(
                profile=profile,
                document=document,
            )
        ]

        if not candidates:
            raise AdapterNotFoundError(
                "没有找到兼容的 "
                "Symbol Extractor Adapter: "
                f"{document.relative_path}"
            )

        candidates.sort(
            key=lambda item: (
                -item.priority,
                item.adapter_id,
            )
        )

        highest_priority = candidates[0].priority

        highest = [item for item in candidates if (item.priority == highest_priority)]

        if len(highest) > 1:
            adapter_ids = ", ".join(item.adapter_id for item in highest)

            raise AmbiguousAdapterError(
                "存在多个相同优先级的 " "Symbol Extractor Adapter: " + adapter_ids
            )

        return candidates[0]

    def list_adapters(
        self,
    ) -> list[SymbolExtractorAdapter]:

        return sorted(
            self._adapters.values(),
            key=lambda item: (
                -item.priority,
                item.adapter_id,
            ),
        )
