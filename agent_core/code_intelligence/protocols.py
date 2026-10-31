from typing import Protocol

from agent_core.code_intelligence.model import (
    SourceDocument,
    SymbolExtractionResult,
    TechnologyProfile,
)
from agent_core.code_intelligence.model import (
    ImportBindingExtractionResult,
    SourceDocument,
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


class ImportBindingExtractorAdapter(Protocol):
    """
    Module Import Binding Extraction。

    和 Symbol Extraction 分开定义，
    防止形成 Fat Interface。
    """

    @property
    def adapter_id(
        self,
    ) -> str: ...

    def supports(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> bool: ...

    def extract_import_bindings(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> ImportBindingExtractionResult: ...


class SymbolReferenceExtractorAdapter(Protocol):
    """
    Source
        ↓
    Reference Occurrence[]

    不负责 Definition Resolution。
    """

    @property
    def adapter_id(
        self,
    ) -> str: ...

    def supports(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> bool: ...

    def extract_references(
        self,
        *,
        profile: TechnologyProfile,
        document: SourceDocument,
    ) -> SymbolReferenceExtractionResult: ...
