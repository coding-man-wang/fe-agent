from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from agent_core.codebase.graph.model import (
    RepositoryDependencyGraph,
)
from agent_core.codebase.graph.query import (
    RepositoryDependencyGraphQuery,
)
from agent_core.codebase.model import (
    RepositoryFileIndex,
)

ImpactFileKind = Literal[
    "production",
    "test",
]

ImpactLevel = Literal[
    "low",
    "medium",
    "high",
    "unknown",
]


class ImpactedFile(BaseModel):
    """
    一个可能受到本次修改影响的文件。

    source_changed_file:
        是哪个被修改文件导致它进入影响范围。

    depth:
        1 = 直接依赖修改文件
        2 = 依赖一个已经受影响的文件

    via_path:
        通过哪个文件传播过来的。
    """

    model_config = ConfigDict(extra="forbid")

    relative_path: str = Field(min_length=1)

    kind: ImpactFileKind

    category: str

    depth: int = Field(ge=1)

    source_changed_file: str = Field(min_length=1)

    via_path: str = Field(min_length=1)

    reason: str = Field(min_length=1)


class ChangeImpactReport(BaseModel):
    """
    一次代码修改对应的影响分析结果。

    注意：

    这个 Report 只表达：

        “哪些文件值得重点检查”

    它不表达：

        “哪些文件允许修改”
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1

    repository_revision: str = Field(min_length=1)

    changed_files: list[str] = Field(min_length=1)

    affected_production_files: list[ImpactedFile] = Field(default_factory=list)

    affected_test_files: list[ImpactedFile] = Field(default_factory=list)

    # Graph 找到的测试 +
    # Change Surface 已知测试。
    #
    # 后面 v1.2.4 会用它参与
    # Verification 范围判断。
    suggested_test_files: list[str] = Field(default_factory=list)

    # 如果某个被修改文件根本不在 Graph 中，
    # 不能武断地说“没有影响”。
    unresolved_changed_files: list[str] = Field(default_factory=list)

    impact_level: ImpactLevel

    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_paths(
        self,
    ) -> "ChangeImpactReport":

        changed = self.changed_files

        if len(changed) != len(set(changed)):
            raise ValueError("changed_files 存在重复路径")

        production_paths = [
            item.relative_path for item in self.affected_production_files
        ]

        test_paths = [item.relative_path for item in self.affected_test_files]

        if len(production_paths) != len(set(production_paths)):
            raise ValueError("affected_production_files " "存在重复路径")

        if len(test_paths) != len(set(test_paths)):
            raise ValueError("affected_test_files " "存在重复路径")

        overlap = set(production_paths) & set(test_paths)

        if overlap:
            raise ValueError(
                "同一个影响文件不能同时是 "
                "production 和 test: " + ", ".join(sorted(overlap))
            )

        return self


class RepositoryChangeImpactAnalyzer:
    """
    根据 Repository Dependency Graph
    判断修改文件可能影响哪些文件。

    当前策略：

        changed file
            ↓
        查 dependents
            ↓
        最多向上查 max_depth 层
            ↓
        production / test 分开记录

    注意：

    这里不修改 Change Surface，
    不提供新的写权限。
    """

    def __init__(
        self,
        *,
        max_depth: int = 2,
    ) -> None:

        if max_depth < 1:
            raise ValueError("max_depth 必须 >= 1")

        self.max_depth = max_depth

    def analyze(
        self,
        *,
        changed_files: list[str],
        graph: RepositoryDependencyGraph,
        index: RepositoryFileIndex,
        repository_revision: str,
        known_test_files: list[str] | None = None,
    ) -> ChangeImpactReport:

        revision = repository_revision.strip()

        if not revision:
            raise ValueError("repository_revision 不能为空")

        normalized_changed = self._normalize_paths(changed_files)

        if not normalized_changed:
            raise ValueError("changed_files 不能为空")

        known_tests = self._normalize_paths(known_test_files or [])

        changed_set = set(normalized_changed)

        node_paths = {node.relative_path for node in graph.nodes}

        file_by_path = {record.relative_path: record for record in index.files}

        graph_query = RepositoryDependencyGraphQuery(graph)

        # key = impacted relative_path
        #
        # 一个文件可能同时受到多个
        # changed file 影响。
        #
        # 最终保留：
        # 1. depth 更浅
        # 2. source path 更稳定
        # 的那一条。
        impacts: dict[
            str,
            ImpactedFile,
        ] = {}

        unresolved_changed_files: list[str] = []

        for changed_path in normalized_changed:

            if changed_path not in node_paths:

                # 新创建的测试文件
                # 本来就不可能出现在
                # 修改前 Dependency Graph。
                #
                # 如果 Change Surface
                # 已经明确知道它是测试，
                # 不应因此把 Impact
                # 误判成 unknown。
                if changed_path in known_tests:
                    continue

                unresolved_changed_files.append(changed_path)

                continue

            hits = graph_query.dependents(
                changed_path,
                max_depth=self.max_depth,
            )

            for hit in hits:

                impacted_path = hit.relative_path

                # 本次自己已经修改的文件，
                # 不再重复当成“受影响文件”。
                if impacted_path in changed_set:
                    continue

                record = file_by_path.get(impacted_path)

                if record is None:
                    continue

                kind: ImpactFileKind = (
                    "test" if record.category == "test" else "production"
                )

                candidate = ImpactedFile(
                    relative_path=(impacted_path),
                    kind=kind,
                    category=record.category,
                    depth=hit.depth,
                    source_changed_file=(changed_path),
                    via_path=(hit.via_path),
                    reason=(
                        self._build_reason(
                            changed_path=(changed_path),
                            impacted_path=(impacted_path),
                            via_path=(hit.via_path),
                            depth=hit.depth,
                        )
                    ),
                )

                existing = impacts.get(impacted_path)

                if existing is None:
                    impacts[impacted_path] = candidate
                    continue

                if self._impact_sort_key(candidate) < self._impact_sort_key(existing):
                    impacts[impacted_path] = candidate

        ordered_impacts = sorted(
            impacts.values(),
            key=self._impact_sort_key,
        )

        production_impacts = [
            item for item in ordered_impacts if item.kind == "production"
        ]

        test_impacts = [item for item in ordered_impacts if item.kind == "test"]

        # Graph 能发现：
        #
        # tests/unit/usersStore.test.ts
        #
        # 这种直接 import production code
        # 的测试。
        #
        # 但是 E2E 测试通常不会 import
        # UsersView，因此 Graph 不一定发现。
        #
        # 所以把 Change Surface 已经知道的
        # tests 也一起保留。
        suggested_test_files = self._normalize_paths(
            [
                *[item.relative_path for item in test_impacts],
                *known_tests,
            ]
        )

        impact_level, level_reasons = self._calculate_impact_level(
            changed_files=(normalized_changed),
            production_impacts=(production_impacts),
            test_impacts=(test_impacts),
            unresolved_changed_files=(unresolved_changed_files),
        )

        return ChangeImpactReport(
            repository_revision=revision,
            changed_files=(normalized_changed),
            affected_production_files=(production_impacts),
            affected_test_files=(test_impacts),
            suggested_test_files=(suggested_test_files),
            unresolved_changed_files=(sorted(unresolved_changed_files)),
            impact_level=impact_level,
            reasons=level_reasons,
        )

    @staticmethod
    def _impact_sort_key(
        item: ImpactedFile,
    ) -> tuple[
        int,
        str,
        str,
        str,
    ]:
        return (
            item.depth,
            item.source_changed_file,
            item.via_path,
            item.relative_path,
        )

    @staticmethod
    def _build_reason(
        *,
        changed_path: str,
        impacted_path: str,
        via_path: str,
        depth: int,
    ) -> str:

        if depth == 1:
            return f"{impacted_path} 直接依赖 " f"{changed_path}"

        return (
            f"{impacted_path} 通过 "
            f"{via_path} 间接受到 "
            f"{changed_path} 的影响，"
            f"depth={depth}"
        )

    @staticmethod
    def _calculate_impact_level(
        *,
        changed_files: list[str],
        production_impacts: list[ImpactedFile],
        test_impacts: list[ImpactedFile],
        unresolved_changed_files: list[str],
    ) -> tuple[
        ImpactLevel,
        list[str],
    ]:

        # 无法分析某个真正修改文件时，
        # 绝不能错误给出 low。
        if unresolved_changed_files:
            return (
                "unknown",
                [
                    (
                        "存在无法进入 Dependency "
                        "Graph 的修改文件，影响范围"
                        "无法完整确认: " + ", ".join(sorted(unresolved_changed_files))
                    )
                ],
            )

        production_count = len(production_impacts)

        test_count = len(test_impacts)

        # 这只是“影响范围大小”，
        # 不是 Bug 概率。
        if production_count >= 5 or len(changed_files) >= 3:
            return (
                "high",
                [
                    (
                        "修改范围较大："
                        f"changed={len(changed_files)}, "
                        f"production_impacts="
                        f"{production_count}"
                    )
                ],
            )

        if production_count > 0 or test_count >= 2:
            return (
                "medium",
                [
                    (
                        "存在需要重点检查的"
                        "生产代码或多个测试："
                        f"production_impacts="
                        f"{production_count}, "
                        f"test_impacts="
                        f"{test_count}"
                    )
                ],
            )

        return (
            "low",
            [
                (
                    "当前 Dependency Graph 中"
                    "影响范围较小："
                    f"production_impacts="
                    f"{production_count}, "
                    f"test_impacts="
                    f"{test_count}"
                )
            ],
        )

    @staticmethod
    def _normalize_paths(
        paths: list[str],
    ) -> list[str]:

        result: list[str] = []

        seen: set[str] = set()

        for path in paths:

            normalized = path.replace(
                "\\",
                "/",
            ).strip()

            if not normalized or normalized in seen:
                continue

            seen.add(normalized)

            result.append(normalized)

        return result
