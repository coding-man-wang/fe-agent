from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from agent_core.code_intelligence.model import (
    CodeSymbol,
    SymbolExtractionResult,
)


class DuplicateSymbolIdError(ValueError):
    pass


class SymbolIndex:
    """
    Repository-level Symbol Index。

    当前只负责：

        Symbol Storage
        +
        Exact Lookup

    暂时不负责：

        Ranking
        Semantic Retrieval
        Reference Resolution
        Call Graph

    后续模块通过该 Index 使用 Symbol，
    不直接访问 Tree-sitter AST。
    """

    def __init__(
        self,
    ) -> None:

        self._by_id: dict[
            str,
            CodeSymbol,
        ] = {}

        self._by_name: dict[
            str,
            list[CodeSymbol],
        ] = defaultdict(list)

        self._by_file: dict[
            str,
            list[CodeSymbol],
        ] = defaultdict(list)

    @classmethod
    def build(
        cls,
        results: Iterable[SymbolExtractionResult],
    ) -> "SymbolIndex":

        index = cls()

        for result in results:
            index.add_result(result)

        return index

    def add_result(
        self,
        result: SymbolExtractionResult,
    ) -> None:

        for symbol in result.symbols:
            self.add_symbol(symbol)

    def add_symbol(
        self,
        symbol: CodeSymbol,
    ) -> None:

        if symbol.symbol_id in self._by_id:
            raise DuplicateSymbolIdError("重复 Symbol ID: " + symbol.symbol_id)

        self._by_id[symbol.symbol_id] = symbol

        self._by_name[symbol.name].append(symbol)

        self._by_file[symbol.relative_path].append(symbol)

    def get_symbol(
        self,
        symbol_id: str,
    ) -> CodeSymbol | None:

        return self._by_id.get(symbol_id)

    def find_by_name(
        self,
        name: str,
    ) -> list[CodeSymbol]:

        return list(self._by_name.get(name, []))

    def find_by_file(
        self,
        relative_path: str,
    ) -> list[CodeSymbol]:

        return list(self._by_file.get(relative_path, []))

    def all_symbols(
        self,
    ) -> list[CodeSymbol]:

        return list(self._by_id.values())

    def __len__(
        self,
    ) -> int:

        return len(self._by_id)
