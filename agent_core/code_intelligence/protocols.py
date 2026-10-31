from typing import Protocol

from agent_core.code_intelligence.model import (
    SourceDocument,
    SymbolExtractionResult,
    TechnologyProfile,
)


class SymbolExtractorAdapter(Protocol):
    """
    Symbol Extraction Technology Adapter。

    Core 只依赖这个 Contract。

    未来实现例如：

        TypeScriptSymbolExtractor
        VueSymbolExtractor
        ReactSymbolExtractor

    但 Symbol Index /
    Retrieval /
    Impact Analysis
    都不应该依赖这些具体类。
    """

    @property
    def adapter_id(
        self,
    ) -> str: ...

    @property
    def priority(
        self,
    ) -> int: ...

    def supports(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> bool: ...

    def extract_symbols(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> SymbolExtractionResult: ...
