from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from agent_core.code_intelligence.model import (
    ModuleExportBinding,
    ModuleExportExtractionResult,
)


class ModuleExportTable:
    """
    Repository Module Export Table。

    SymbolIndex:
        Symbol 定义在哪里？

    ModuleExportTable:
        Module 对外暴露什么名字？
    """

    def __init__(
        self,
    ) -> None:

        self._by_file: dict[
            str,
            list[ModuleExportBinding],
        ] = defaultdict(list)

    @classmethod
    def build(
        cls,
        results: Iterable[ModuleExportExtractionResult],
    ) -> "ModuleExportTable":

        table = cls()

        for result in results:
            table.add_result(result)

        return table

    def add_result(
        self,
        result: ModuleExportExtractionResult,
    ) -> None:

        self._by_file[result.relative_path].extend(result.exports)

    def find_explicit(
        self,
        *,
        relative_path: str,
        exported_name: str,
    ) -> list[ModuleExportBinding]:

        return [
            item
            for item in self._by_file.get(
                relative_path,
                [],
            )
            if (
                item.binding_kind != "star_reexport"
                and item.exported_name == exported_name
            )
        ]

    def find_star_exports(
        self,
        relative_path: str,
    ) -> list[ModuleExportBinding]:

        return [
            item
            for item in self._by_file.get(
                relative_path,
                [],
            )
            if (item.binding_kind == "star_reexport")
        ]

    def find_by_file(
        self,
        relative_path: str,
    ) -> list[ModuleExportBinding]:

        return list(
            self._by_file.get(
                relative_path,
                [],
            )
        )
