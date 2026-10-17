from pathlib import PurePosixPath

from agent_core.codebase.model import (
    RepositoryFileIndex,
    HybridSearchResponse,
    HybridRelevantFileSelection,
    HybridRelevantFile,
)
from agent_core.sdd.model import TaskDefinition
from agent_core.codebase.retrieval_query import (
    RetrievalQuery,
)


class HybridRelevantFileSelector:
    CATEGORY_BONUS = {
        "code_change": {
            "source": 30,
            "config": 15,
            "test": 10,
            "documentation": 5,
        },
        "test": {
            "test": 30,
            "source": 20,
            "config": 10,
            "documentation": 5,
        },
        "investigation": {
            "source": 20,
            "config": 20,
            "documentation": 15,
            "test": 10,
        },
        "verification": {
            "test": 25,
            "config": 20,
            "source": 15,
            "documentation": 5,
        },
    }

    def __init__(
        self,
        *,
        query_rrf_k: int = 60,
        selection_rrf_k: int = 60,
        retrieval_weight: float = 1.0,
        policy_weight: float = 0.35,
    ) -> None:

        if query_rrf_k < 1:
            raise ValueError("query_rrf_k 必须 >= 1")

        if selection_rrf_k < 1:
            raise ValueError("selection_rrf_k 必须 >= 1")

        if retrieval_weight < 0:
            raise ValueError("retrieval_weight 不能小于 0")

        if policy_weight < 0:
            raise ValueError("policy_weight 不能小于 0")

        self.query_rrf_k = query_rrf_k
        self.selection_rrf_k = selection_rrf_k

        self.retrieval_weight = retrieval_weight

        self.policy_weight = policy_weight

    def select(
        self,
        *,
        index: RepositoryFileIndex,
        hybrid_responses: list[HybridSearchResponse],
        task: TaskDefinition,
        query_specs: list[RetrievalQuery] | None = None,
        max_files: int = 8,
        max_total_bytes: int = 200_000,
    ) -> HybridRelevantFileSelection:
        if max_files < 1:
            raise ValueError("max_files 必须 >= 1")

        if max_total_bytes < 1:
            raise ValueError("max_total_bytes 必须 >= 1")

        file_map = {file.relative_path: file for file in index.files}

        aggregated = self._aggregate_retrieval(
            hybrid_responses,
            query_specs=query_specs or [],
        )

        candidate_count = len(aggregated)

        skipped_non_indexable: list[str] = []

        eligible: dict[str, dict] = {}

        for path, data in aggregated.items():
            file_record = file_map.get(path)

            if file_record is None:
                continue

            if not file_record.is_indexable:
                skipped_non_indexable.append(path)

                continue

            policy_score, policy_reasons = self._calculate_policy_score(
                path=path,
                category=(file_record.category),
                task=task,
            )

            eligible[path] = {
                **data,
                "file_record": file_record,
                "policy_score": (policy_score),
                "policy_reasons": (policy_reasons),
            }

        self._assign_retrieval_ranks(eligible)

        self._assign_policy_ranks(eligible)

        ranked = self._build_final_ranking(eligible)

        selected: list[HybridRelevantFile] = []

        skipped_by_budget: list[str] = []

        current_bytes = 0

        for file in ranked:
            if len(selected) >= max_files:
                skipped_by_budget.append(file.relative_path)
                continue

            if current_bytes + file.size_bytes > max_total_bytes:
                skipped_by_budget.append(file.relative_path)
                continue

            selected.append(file)

            current_bytes += file.size_bytes

        return HybridRelevantFileSelection(
            candidate_count=(candidate_count),
            eligible_count=len(eligible),
            selected_count=len(selected),
            total_selected_bytes=(current_bytes),
            selected_files=selected,
            skipped_by_budget=(skipped_by_budget),
            skipped_non_indexable=(skipped_non_indexable),
        )

    def _aggregate_retrieval(
        self,
        responses: list[HybridSearchResponse],
        *,
        query_specs: list[RetrievalQuery],
    ) -> dict[str, dict]:
        intent_map: dict[
            str,
            set[str],
        ] = {}

        for query_spec in query_specs:
            intents = intent_map.setdefault(
                query_spec.text,
                set(),
            )

            intents.update(query_spec.intents)

        aggregated: dict[
            str,
            dict,
        ] = {}

        for response in responses:

            for candidate in response.candidates:

                data = aggregated.setdefault(
                    candidate.relative_path,
                    {
                        "retrieval_score": 0.0,
                        "queries": set(),
                        "intents": set(),
                        "reasons": [],
                        "best_semantic_chunk": None,
                        "best_semantic_score": None,
                        "lexical_hit_count": 0,
                        "semantic_hit_count": 0,
                        "hybrid_hit_count": 0,
                    },
                )

                # -----------------------------
                # 关键变化：
                #
                # 不再重新只根据 rank
                # 计算一次 RRF。
                #
                # 直接保留 HybridRetriever
                # 已经计算好的 fusion_score。
                # -----------------------------

                data["retrieval_score"] += candidate.fusion_score

                data["queries"].add(response.query)

                data["intents"].update(
                    intent_map.get(
                        response.query,
                        set(),
                    )
                )
                # -----------------------------
                # 记录搜索证据来自哪里。
                # -----------------------------

                has_lexical = candidate.lexical_rank is not None

                has_semantic = candidate.semantic_rank is not None

                if has_lexical:
                    data["lexical_hit_count"] += 1

                if has_semantic:
                    data["semantic_hit_count"] += 1

                if has_lexical and has_semantic:
                    data["hybrid_hit_count"] += 1

                # -----------------------------
                # 保留最强 Semantic Evidence。
                # -----------------------------

                if candidate.semantic_score is not None:

                    current_best = data["best_semantic_score"]

                    if current_best is None or candidate.semantic_score > current_best:
                        data["best_semantic_score"] = candidate.semantic_score

                        data["best_semantic_chunk"] = candidate.best_semantic_chunk

                # -----------------------------
                # Diagnostics
                # -----------------------------

                evidence = []

                if has_lexical:
                    evidence.append("lexical")

                if has_semantic:
                    evidence.append("semantic")

                data["reasons"].append(
                    (
                        "Hybrid Retrieval "
                        f"query={response.query}, "
                        f"signals="
                        f"{'+'.join(evidence)}, "
                        f"fusion_score="
                        f"{candidate.fusion_score:.6f}"
                    )
                )

        return aggregated

    @staticmethod
    def _calculate_target_bonus(
        *,
        path: str,
        target: str,
    ) -> tuple[int, str | None]:
        """
        target_modules 的信息精确程度不同，
        不能统一 +25。

        精确文件：
            +25

        文件名 / Symbol：
            +15

        大目录：
            +5
        """

        normalized_path = (
            path.replace(
                "\\",
                "/",
            )
            .lower()
            .strip("/")
        )

        normalized_target = (
            target.replace(
                "\\",
                "/",
            )
            .lower()
            .strip("/")
        )

        if not normalized_target:
            return (
                0,
                None,
            )

        target_path = PurePosixPath(normalized_target)

        target_name = target_path.name

        path_name = PurePosixPath(normalized_path).name

        # ---------------------------------
        # 情况一：
        #
        # 明确文件路径 / 文件名。
        #
        # 例如：
        #
        # src/components/UserProfileForm.vue
        # ---------------------------------

        if target_path.suffix:

            if (
                normalized_path == normalized_target
                or normalized_path.endswith("/" + normalized_target)
                or path_name == target_name
            ):
                return (
                    25,
                    ("精确匹配 " "target file: " f"{target}"),
                )

            return (
                0,
                None,
            )

        # ---------------------------------
        # 情况二：
        #
        # 看起来是目录。
        #
        # src/components
        # tests/unit
        #
        # 这种信息很粗，
        # 只能轻微加分。
        # ---------------------------------

        if "/" in normalized_target:

            directory_prefix = normalized_target.rstrip("/") + "/"

            if normalized_path.startswith(directory_prefix):
                return (
                    5,
                    ("匹配 target directory: " f"{target}"),
                )

            return (
                0,
                None,
            )

        # ---------------------------------
        # 情况三：
        #
        # 文件名 / Symbol 名。
        #
        # UserProfileForm
        # ---------------------------------

        if normalized_target in path_name:
            return (
                15,
                ("匹配 target symbol: " f"{target}"),
            )

        return (
            0,
            None,
        )

    def _calculate_policy_score(
        self,
        *,
        path: str,
        category: str,
        task: TaskDefinition,
    ) -> tuple[int, list[str]]:
        score = 0

        reasons: list[str] = []

        category_bonus = self.CATEGORY_BONUS.get(
            task.task_type,
            {},
        ).get(
            category,
            0,
        )

        if category_bonus:
            score += category_bonus

            reasons.append(
                f"{task.task_type} Task " f"对 {category} 文件 " f"+{category_bonus}"
            )

        normalized_path = path.lower()

        best_target_bonus = 0

        best_target_reason = None

        for target in task.target_modules:

            (
                target_bonus,
                target_reason,
            ) = self._calculate_target_bonus(
                path=path,
                target=target,
            )

            if target_bonus > best_target_bonus:
                best_target_bonus = target_bonus

                best_target_reason = target_reason

        if best_target_bonus:

            score += best_target_bonus

            if best_target_reason:
                reasons.append(best_target_reason)

        return (
            score,
            reasons,
        )

    def _assign_retrieval_ranks(
        self,
        eligible: dict[str, dict],
    ) -> None:

        ordered = sorted(
            eligible.items(),
            key=lambda item: (
                -item[1]["retrieval_score"],
                item[0],
            ),
        )

        for rank, (
            _,
            data,
        ) in enumerate(
            ordered,
            start=1,
        ):
            data["retrieval_rank"] = rank

    def _assign_policy_ranks(
        self,
        eligible: dict[str, dict],
    ) -> None:

        ordered = sorted(
            [item for item in eligible.items() if (item[1]["policy_score"] > 0)],
            key=lambda item: (
                -item[1]["policy_score"],
                item[0],
            ),
        )

        for rank, (
            _,
            data,
        ) in enumerate(
            ordered,
            start=1,
        ):
            data["policy_rank"] = rank

    def _build_final_ranking(
        self,
        eligible: dict[str, dict],
    ) -> list[HybridRelevantFile]:

        results: list[HybridRelevantFile] = []

        for path, data in eligible.items():

            retrieval_rank = data["retrieval_rank"]

            policy_rank = data.get("policy_rank")

            selection_score = self.retrieval_weight / (
                self.selection_rrf_k + retrieval_rank
            )

            if policy_rank is not None:
                selection_score += self.policy_weight / (
                    self.selection_rrf_k + policy_rank
                )

            file_record = data["file_record"]

            reasons = list(data["reasons"])

            reasons.extend(data["policy_reasons"])

            results.append(
                HybridRelevantFile(
                    relative_path=path,
                    category=file_record.category,
                    language=file_record.language,
                    size_bytes=file_record.size_bytes,
                    retrieval_score=data["retrieval_score"],
                    retrieval_rank=retrieval_rank,
                    policy_score=data["policy_score"],
                    policy_rank=policy_rank,
                    selection_score=selection_score,
                    best_semantic_chunk=data["best_semantic_chunk"],
                    best_semantic_score=data["best_semantic_score"],
                    query_hit_count=len(data["queries"]),
                    lexical_hit_count=data["lexical_hit_count"],
                    semantic_hit_count=data["semantic_hit_count"],
                    hybrid_hit_count=data["hybrid_hit_count"],
                    matched_queries=sorted(data["queries"]),
                    matched_intents=sorted(data["intents"]),
                    reasons=reasons,
                )
            )

        results.sort(
            key=lambda item: (
                -item.selection_score,
                item.relative_path,
            )
        )

        return results
