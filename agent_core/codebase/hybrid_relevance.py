from agent_core.codebase.model import (
    RepositoryFileIndex,
    HybridSearchResponse,
    HybridRelevantFileSelection,
    HybridRelevantFile,
)
from agent_core.sdd.model import TaskDefinition


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
            max_files: int = 8,
            max_total_bytes: int = 200_000,
    ) -> HybridRelevantFileSelection:
        if max_files < 1:
            raise ValueError("max_files 必须 >= 1")

        if max_total_bytes < 1:
            raise ValueError("max_total_bytes 必须 >= 1")

        file_map = {file.relative_path: file for file in index.files}

        aggregated = self._aggregate_retrieval(hybrid_responses)

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
    ) -> dict[str, dict]:
        aggregated: dict[str, dict] = {}

        for response in responses:
            for rank, candidate in enumerate(response.candidates, start=1):
                data = aggregated.setdefault(
                    candidate.relative_path,
                    {
                        "retrieval_score": 0.0,
                        "queries": set(),
                        "reasons": [],
                        "best_semantic_chunk": None,
                        "best_semantic_score": None,
                    },
                )

                data["retrieval_score"] += 1.0 / (self.query_rrf_k + rank)

                data["queries"].add(response.query)

                data["reasons"].append(
                    "Hybrid Retrieval " f"query={response.query}, " f"rank={rank}"
                )

                if candidate.semantic_score is not None:
                    current_best = data["best_semantic_score"]

                    if current_best is None or candidate.semantic_score > current_best:
                        data["best_semantic_score"] = candidate.semantic_score

                        data["best_semantic_chunk"] = candidate.best_semantic_chunk

        return aggregated

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

        matched_targets: list[str] = []

        for target in task.target_modules:
            normalized_target = target.strip().lower().replace("\\", "/")

            if not normalized_target:
                continue

            if normalized_target in normalized_path:
                matched_targets.append(target)

        if matched_targets:
            score += 25

            reasons.append("匹配 target_modules: " + ", ".join(matched_targets))

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
                    category=(file_record.category),
                    language=(file_record.language),
                    size_bytes=(file_record.size_bytes),
                    retrieval_score=(data["retrieval_score"]),
                    retrieval_rank=(retrieval_rank),
                    policy_score=(data["policy_score"]),
                    policy_rank=(policy_rank),
                    selection_score=(selection_score),
                    matched_queries=sorted(data["queries"]),
                    best_semantic_chunk=(data["best_semantic_chunk"]),
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
