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

    def find_exported_by_file_and_name(
        self,
        *,
        relative_path: str,
        name: str,
    ) -> list[CodeSymbol]:
        """
        查询某 Module 真正导出的 Symbol。

        注意：

        exported 的语义已经在
        TypeScript Adapter 中限制在
        Module Export Boundary。
        """

        return [
            symbol
            for symbol in self.find_by_file(relative_path)
            if (symbol.name == name and symbol.exported)
        ]

    def file_paths(
        self,
    ) -> set[str]:

        return set(self._by_file.keys())

    def find_by_file_and_name(
        self,
        *,
        relative_path: str,
        name: str,
    ) -> list[CodeSymbol]:
        """
        查找 Module 内真实定义的 Symbol。

        注意：

        不要求 symbol.exported == True。

        因为：

            function foo() {}

            export { foo }

        foo 本身定义时并不是
        export declaration，
        但 Module Export Table
        会把它暴露出去。
        """

        return [
            symbol for symbol in self.find_by_file(relative_path) if symbol.name == name
        ]

    def find_containing(
        self,
        *,
        relative_path: str,
        line_number: int,
    ) -> list[CodeSymbol]:
        """
        查找包含该源码行的所有 Symbol。

        从最内层 Symbol 到最外层 Symbol 排序。
        """

        result = [
            symbol
            for symbol in self.find_by_file(relative_path)
            if (symbol.start_line <= line_number <= symbol.end_line)
        ]

        result.sort(
            key=lambda symbol: (
                (symbol.end_line - symbol.start_line),
                -symbol.start_line,
            )
        )

        return result

    def find_innermost_containing(
        self,
        *,
        relative_path: str,
        line_number: int,
    ) -> CodeSymbol | None:

        candidates = self.find_containing(
            relative_path=(relative_path),
            line_number=(line_number),
        )

        if not candidates:
            return None

        return candidates[0]

    def parent_chain(
        self,
        symbol_id: str,
    ) -> list[CodeSymbol]:
        """
        返回：

            current parent
            grand parent
            ...

        不包含当前 Symbol 自己。
        """

        result: list[CodeSymbol] = []

        current = self.get_symbol(symbol_id)

        visited: set[str] = set()

        while current is not None and (current.parent_symbol_id is not None):

            parent_id = current.parent_symbol_id

            if parent_id in visited:
                break

            visited.add(parent_id)

            parent = self.get_symbol(parent_id)

            if parent is None:
                break

            result.append(parent)

            current = parent

        return result
