from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agent_core.codebase.graph.model import (
    RepositoryDependencyGraph,
)
from agent_core.codebase.graph.query import (
    RepositoryDependencyGraphQuery,
)
from agent_core.codebase.model import (
    FileRecord,
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryFileIndex,
)

GraphRelation = Literal[
    "dependency",
    "dependent",
]


@dataclass
class _GraphCandidate:
    """
    Graph Context Expansion 内部候选。

    这里不直接复用 HybridRelevantFile，
    因为一个 Graph 文件可能通过多个 anchor、
    多条 dependency path 被发现。

    我们先在这一层完成：
    1. 去重
    2. 最优路径选择
    3. 确定性排序
    4. Graph evidence 聚合

    最后再转换为 HybridRelevantFile。
    """

    relative_path: str

    relation: GraphRelation

    depth: int

    anchor_path: str

    anchor_rank: int

    anchor_position: int

    via_path: str

    reasons: list[str] = field(default_factory=list)

    @property
    def relation_priority(self) -> int:
        """
        同一 depth 下：

        dependency 优先于 dependent。

        例如：

        depth=1 dependency
        depth=1 dependent
        depth=2 dependency
        depth=2 dependent

        这是一个非常明确的 deterministic policy。
        """

        if self.relation == "dependency":
            return 0

        return 1

    @property
    def sort_key(
        self,
    ) -> tuple[int, int, int, int, str]:
        """
        Graph Candidate 的稳定排序规则。

        优先级：

        1. depth 越浅越优先
        2. dependency 优先于 dependent
        3. anchor retrieval_rank 越高越优先
        4. anchor 在当前 selection 中的位置
        5. relative_path 保证最终 deterministic

        最终效果：

        dependency depth=1
        dependent  depth=1
        dependency depth=2
        dependent  depth=2
        """

        return (
            self.depth,
            self.relation_priority,
            self.anchor_rank,
            self.anchor_position,
            self.relative_path,
        )


class RepositoryGraphContextExpander:
    """
    使用 Repository Dependency Graph
    对 Retrieval Selection 做结构上下文扩展。

    核心边界：

        Graph Edge != Modification Permission

    Graph 只决定：

        "Agent 还应该阅读哪些文件"

    Graph 不决定：

        "Agent 可以修改哪些文件"

    修改权限仍然必须经过：

        Investigation
            ->
        candidate_target evidence
            ->
        ChangeSurfaceResolver
            ->
        implementation_targets

    v1.2.2-A Policy：

        最终 max_files 个 Context Slot

        Retrieval Anchors
            +
        Graph Expansion

    默认：

        max_files=8
        max_graph_files=3

    即：

        最多使用 3 个 Context Slot
        放 Graph 新发现的文件。

    同时保护 target_match_type != none
    的 Retrieval 文件，避免 Graph 把 Planner
    已经明确指向的 Target Evidence 挤出 Context。
    """

    REPLACEABLE_CONTEXT_ROLES = frozenset(
        {
            "reference",
            "unclassified",
        }
    )

    def __init__(
        self,
        *,
        max_files: int = 8,
        max_graph_files: int = 3,
        min_retrieval_files: int = 5,
        dependency_depth: int = 2,
        dependent_depth: int = 1,
    ) -> None:
        if max_files < 1:
            raise ValueError("max_files 必须 >= 1")

        if max_graph_files < 0:
            raise ValueError("max_graph_files 必须 >= 0")

        if dependency_depth < 1:
            raise ValueError("dependency_depth 必须 >= 1")

        if dependent_depth < 1:
            raise ValueError("dependent_depth 必须 >= 1")

        if min_retrieval_files < 0:
            raise ValueError("min_retrieval_files 必须 >= 0")

        if min_retrieval_files > max_files:
            raise ValueError("min_retrieval_files 不能大于 max_files")

        self.min_retrieval_files = min_retrieval_files

        self.max_files = max_files

        self.max_graph_files = min(
            max_graph_files,
            max_files,
        )

        self.dependency_depth = dependency_depth

        self.dependent_depth = dependent_depth

    def expand(
        self,
        *,
        selection: HybridRelevantFileSelection,
        index: RepositoryFileIndex,
        graph: RepositoryDependencyGraph,
    ) -> HybridRelevantFileSelection:
        """
        使用 Dependency Graph 扩展当前 Retrieval Selection。

        整体流程：

        1. 对 Retrieval Selection 去重
        2. 以 selected_files 作为 Graph anchors
        3. 查询 dependencies
        4. 查询 dependents
        5. 排除已经存在于 Retrieval 的文件
        6. 排除不可进入 Context 的文件
        7. 对 Graph Candidates 做 deterministic ranking
        8. 根据 Context Budget 合并 Retrieval + Graph
        9. 返回新的 HybridRelevantFileSelection
        """

        base_files = self._deduplicate_base_files(selection.selected_files)

        file_records = {item.relative_path: item for item in index.files}

        existing_paths = {item.relative_path for item in base_files}

        graph_query = RepositoryDependencyGraphQuery(graph)

        (
            graph_candidates,
            graph_skipped_non_indexable,
        ) = self._collect_graph_candidates(
            base_files=base_files,
            existing_paths=existing_paths,
            file_records=file_records,
            graph_query=graph_query,
        )

        graph_candidates.sort(key=lambda item: item.sort_key)

        graph_budget = self._calculate_graph_budget(
            base_files=base_files,
            graph_candidate_count=len(graph_candidates),
        )

        # ---------------------------------------
        # Graph Candidate 需要真正占用多少个
        # 已有 Retrieval Slot。
        #
        # 如果：
        #
        # base=6
        # max_files=8
        # graph_budget=3
        #
        # 那么：
        #
        # 2 个直接使用空位
        # 只有 1 个需要替换 Retrieval
        # ---------------------------------------

        free_slots = max(
            0,
            self.max_files - len(base_files),
        )

        replacement_count = max(
            0,
            graph_budget - free_slots,
        )

        chosen_base_files = self._choose_base_files(
            base_files=base_files,
            replacement_count=(replacement_count),
        )

        chosen_graph_candidates = graph_candidates[:graph_budget]

        next_retrieval_rank = (
            max(
                (item.retrieval_rank for item in base_files),
                default=0,
            )
            + 1
        )

        graph_files: list[HybridRelevantFile] = []

        for offset, candidate in enumerate(chosen_graph_candidates):
            record = file_records[candidate.relative_path]

            graph_files.append(
                self._build_graph_file(
                    candidate=candidate,
                    record=record,
                    retrieval_rank=(next_retrieval_rank + offset),
                )
            )

        final_files = [
            *chosen_base_files,
            *graph_files,
        ]

        final_paths = {item.relative_path for item in final_files}

        dropped_base_paths = [
            item.relative_path
            for item in base_files
            if item.relative_path not in final_paths
        ]

        dropped_graph_paths = [
            item.relative_path for item in graph_candidates[graph_budget:]
        ]

        skipped_by_budget = self._merge_unique_paths(
            [
                *selection.skipped_by_budget,
                *dropped_base_paths,
                *dropped_graph_paths,
            ],
            exclude=final_paths,
        )

        skipped_non_indexable = self._merge_unique_paths(
            [
                *selection.skipped_non_indexable,
                *graph_skipped_non_indexable,
            ],
            exclude=final_paths,
        )

        return selection.model_copy(
            deep=True,
            update={
                "selected_files": final_files,
                "selected_count": len(final_files),
                "total_selected_bytes": sum(item.size_bytes for item in final_files),
                "skipped_by_budget": (skipped_by_budget),
                "skipped_non_indexable": (skipped_non_indexable),
            },
        )

    def _collect_graph_candidates(
        self,
        *,
        base_files: list[HybridRelevantFile],
        existing_paths: set[str],
        file_records: dict[str, FileRecord],
        graph_query: RepositoryDependencyGraphQuery,
    ) -> tuple[
        list[_GraphCandidate],
        list[str],
    ]:
        """
        从所有 Retrieval Anchor 出发收集 Graph Candidate。

        一个文件可能：

        - 被多个 anchor 引用
        - 同时作为 dependency / dependent 被发现
        - 通过不同 depth 到达

        最终每个 relative_path 只保留一个 Candidate，
        Candidate 主路径选择 sort_key 最优的一条，
        但 reasons 会保留所有发现路径。
        """

        candidates: dict[
            str,
            _GraphCandidate,
        ] = {}

        skipped_non_indexable: list[str] = []

        for anchor_position, anchor in enumerate(base_files):
            dependency_hits = graph_query.dependencies(
                anchor.relative_path,
                max_depth=(self.dependency_depth),
            )

            for hit in dependency_hits:
                self._add_graph_candidate(
                    candidates=candidates,
                    skipped_non_indexable=(skipped_non_indexable),
                    existing_paths=(existing_paths),
                    file_records=file_records,
                    relation="dependency",
                    relative_path=(hit.relative_path),
                    depth=hit.depth,
                    via_path=hit.via_path,
                    anchor=anchor,
                    anchor_position=(anchor_position),
                )

            dependent_hits = graph_query.dependents(
                anchor.relative_path,
                max_depth=(self.dependent_depth),
            )

            for hit in dependent_hits:
                self._add_graph_candidate(
                    candidates=candidates,
                    skipped_non_indexable=(skipped_non_indexable),
                    existing_paths=(existing_paths),
                    file_records=file_records,
                    relation="dependent",
                    relative_path=(hit.relative_path),
                    depth=hit.depth,
                    via_path=hit.via_path,
                    anchor=anchor,
                    anchor_position=(anchor_position),
                )

        return (
            list(candidates.values()),
            self._merge_unique_paths(skipped_non_indexable),
        )

    def _add_graph_candidate(
        self,
        *,
        candidates: dict[
            str,
            _GraphCandidate,
        ],
        skipped_non_indexable: list[str],
        existing_paths: set[str],
        file_records: dict[str, FileRecord],
        relation: GraphRelation,
        relative_path: str,
        depth: int,
        via_path: str,
        anchor: HybridRelevantFile,
        anchor_position: int,
    ) -> None:
        """
        添加一个 Graph Candidate。

        这里集中处理：

        - Retrieval 已存在文件去重
        - FileIndex 存在性检查
        - is_indexable 检查
        - 多路径 candidate 合并
        """

        # Retrieval 已经选择的文件不需要 Graph 再添加。
        if relative_path in existing_paths:
            return

        record = file_records.get(relative_path)

        # Graph 与 FileIndex 理论上应一致，
        # 但这里仍然做 defensive check。
        if record is None or not record.is_indexable:
            skipped_non_indexable.append(relative_path)
            return

        reason = self._build_reason(
            relation=relation,
            anchor_path=(anchor.relative_path),
            relative_path=relative_path,
            via_path=via_path,
            depth=depth,
        )

        incoming = _GraphCandidate(
            relative_path=relative_path,
            relation=relation,
            depth=depth,
            anchor_path=(anchor.relative_path),
            anchor_rank=(anchor.rerank_rank or anchor.retrieval_rank),
            anchor_position=anchor_position,
            via_path=via_path,
            reasons=[reason],
        )

        existing = candidates.get(relative_path)

        if existing is None:
            candidates[relative_path] = incoming
            return

        # 同一个文件可能通过多条 Graph Path 被发现。
        #
        # reasons 全部保留，
        # 但主 Candidate 使用最优路径。
        reasons = self._merge_unique_paths(
            [
                *existing.reasons,
                reason,
            ]
        )

        if incoming.sort_key < existing.sort_key:
            incoming.reasons = reasons

            candidates[relative_path] = incoming
            return

        existing.reasons = reasons

    def _build_protected_paths(
        self,
        *,
        base_files: list[HybridRelevantFile],
    ) -> set[str]:
        """
        建立 Retrieval Evidence Floor。

        保护：

        1. 最强的前 N 个 Retrieval/Rerank Evidence
        2. implementation_target
        3. test
        4. policy
        5. explicit target evidence
        """

        ranked = sorted(
            base_files,
            key=lambda item: (
                self._effective_rank(item),
                item.relative_path,
            ),
        )

        protected = {item.relative_path for item in ranked[: self.min_retrieval_files]}

        for item in base_files:
            if self._is_pinned(item):
                protected.add(item.relative_path)

            if item.context_role in {
                "implementation_target",
                "test",
                "policy",
            }:
                protected.add(item.relative_path)

        return protected

    def _calculate_graph_budget(
        self,
        *,
        base_files: list[HybridRelevantFile],
        graph_candidate_count: int,
    ) -> int:
        """
        计算 Graph Expansion 真正可以使用的 Context Slot。

        新策略：

        1. Context 还有空位：
           Graph 可以直接填充。

        2. Context 已满：
           Graph 只能替换弱 Retrieval Evidence。

        3. 强 Retrieval Evidence：
           - implementation_target
           - dependency
           - test
           - policy
           - target_match_type != none

           不允许仅凭 Graph Relation 被挤掉。

        这使 Graph 从：

            destructive replacement

        变成：

            evidence-preserving enrichment
        """

        free_slots = max(
            0,
            self.max_files - len(base_files),
        )

        protected_paths = self._build_protected_paths(
            base_files=base_files,
        )

        replaceable_count = sum(
            1
            for item in base_files
            if self._is_replaceable(
                item=item,
                protected_paths=protected_paths,
            )
        )

        available_graph_slots = free_slots + replaceable_count

        return min(
            self.max_graph_files,
            graph_candidate_count,
            available_graph_slots,
        )

    def _choose_base_files(
        self,
        *,
        base_files: list[HybridRelevantFile],
        replacement_count: int,
    ) -> list[HybridRelevantFile]:
        """
        根据 Graph Expansion 需要的 Slot，
        从 Retrieval 中淘汰弱 Evidence。

        淘汰规则：

            只淘汰 reference / unclassified

        并且：

            从 Retrieval 排名靠后的文件开始淘汰。

        这样：

            高排名 reference

        仍然优先于：

            低排名 reference。

        原 Retrieval 顺序最终保持不变。
        """

        if replacement_count <= 0:
            return list(base_files)

        protected_paths = self._build_protected_paths(
            base_files=base_files,
        )

        replaceable = [
            item
            for item in base_files
            if self._is_replaceable(
                item=item,
                protected_paths=protected_paths,
            )
        ]

        # 后面的 Retrieval Candidate
        # 优先被淘汰。
        replaceable.sort(
            key=lambda item: (
                self._effective_rank(item),
                item.relative_path,
            ),
            reverse=True,
        )

        drop_paths = {item.relative_path for item in replaceable[:replacement_count]}

        return [item for item in base_files if item.relative_path not in drop_paths]

    @staticmethod
    def _is_pinned(
        item: HybridRelevantFile,
    ) -> bool:
        """
        Planner Target Evidence 不允许被 Graph
        无理由挤出 Context。
        """

        return item.target_match_type != "none"

    @staticmethod
    def _effective_rank(
        item: HybridRelevantFile,
    ) -> int:
        return item.rerank_rank or item.retrieval_rank

    @classmethod
    def _is_replaceable(
        self,
        *,
        item: HybridRelevantFile,
        protected_paths: set[str],
    ) -> bool:
        if item.relative_path in protected_paths:
            return False

        if self._is_pinned(item):
            return False

        if item.context_role in {
            "implementation_target",
            "test",
            "policy",
        }:
            return False

        return True

    @staticmethod
    def _build_graph_file(
        *,
        candidate: _GraphCandidate,
        record: FileRecord,
        retrieval_rank: int,
    ) -> HybridRelevantFile:
        """
        将 Graph Candidate 转换成项目现有的
        HybridRelevantFile。

        注意安全边界：

        retrieval_score = 0
        target_match_type = none
        context_role = dependency

        即 Graph 文件：

        - 不是 Retrieval 命中的结果
        - 不是 Planner 明确修改目标
        - 只是需要进一步阅读的结构上下文
        """

        if candidate.relation == "dependency":
            role_reason = (
                "Repository Dependency Graph "
                "发现该文件是已检索文件的结构依赖；"
                "加入 Context 用于代码调查，"
                "不代表允许修改该文件。"
            )
        else:
            role_reason = (
                "Repository Dependency Graph "
                "发现该文件依赖已检索文件；"
                "加入 Context 用于上游影响分析，"
                "不代表允许修改该文件。"
            )

        return HybridRelevantFile(
            relative_path=(candidate.relative_path),
            category=record.category,
            language=record.language,
            size_bytes=record.size_bytes,
            # Graph Evidence 不是 Retrieval Evidence。
            retrieval_score=0.0,
            retrieval_rank=retrieval_rank,
            policy_score=0,
            policy_rank=None,
            selection_score=0.0,
            best_semantic_chunk=None,
            best_semantic_score=None,
            query_hit_count=0,
            lexical_hit_count=0,
            semantic_hit_count=0,
            hybrid_hit_count=0,
            matched_queries=[],
            matched_intents=[],
            # 非常重要：
            #
            # Graph 找到文件
            # != Planner 明确要求修改文件。
            target_match_type="none",
            target_match_value=None,
            # Graph 文件统一作为结构依赖上下文。
            #
            # dependent 也不是 reference：
            #
            # reference 代表类似实现/参考实现；
            # dependent 代表结构影响关系。
            context_role="dependency",
            role_reason=role_reason,
            rerank_score=None,
            rerank_rank=None,
            reasons=list(candidate.reasons),
        )

    @staticmethod
    def _build_reason(
        *,
        relation: GraphRelation,
        anchor_path: str,
        relative_path: str,
        via_path: str,
        depth: int,
    ) -> str:
        if relation == "dependency":
            return (
                "Graph dependency: "
                f"anchor={anchor_path}, "
                f"target={relative_path}, "
                f"via={via_path}, "
                f"depth={depth}"
            )

        return (
            "Graph dependent: "
            f"anchor={anchor_path}, "
            f"source={relative_path}, "
            f"via={via_path}, "
            f"depth={depth}"
        )

    @staticmethod
    def _deduplicate_base_files(
        files: list[HybridRelevantFile],
    ) -> list[HybridRelevantFile]:
        """
        Defensive deduplication。

        正常 Retrieval Selection 不应该出现重复文件，
        但 Graph Expander 自身不依赖这个隐含假设。
        """

        result: list[HybridRelevantFile] = []

        seen: set[str] = set()

        for item in files:
            if item.relative_path in seen:
                continue

            seen.add(item.relative_path)

            result.append(item)

        return result

    @staticmethod
    def _merge_unique_paths(
        paths: list[str],
        *,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """
        保序去重。

        skipped_by_budget / reasons 等诊断字段
        都希望保持 deterministic。
        """

        excluded = exclude if exclude is not None else set()

        result: list[str] = []

        seen: set[str] = set()

        for path in paths:
            if path in excluded:
                continue

            if path in seen:
                continue

            seen.add(path)

            result.append(path)

        return result
