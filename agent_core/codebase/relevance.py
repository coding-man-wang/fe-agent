from agent_core.codebase.model import (
    ExactSearchResponse,
    RepositoryFileIndex,
    RelevantFile,
    RelevantFileSelection,
)

from agent_core.sdd.model import (
    TaskDefinition,
)


class RelevantFileSelector:
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

    def select(
        self,
        *,
        index: RepositoryFileIndex,
        search_responses: list[ExactSearchResponse],
        task: TaskDefinition,
        max_files: int = 8,
        max_total_bytes: int = 200_000,
    ):
        if max_files < 1:
            raise ValueError("max_files 必须 >= 1")

        if max_total_bytes < 1:
            raise ValueError("max_total_bytes 必须 >= 1")

        file_map = {file.relative_path: file for file in index.files}

        candidates: dict[str, dict] = {}

        for response in search_responses:
            for result in response.results:
                candidate = candidates.setdefault(
                    result.relative_path,
                    {
                        "base_score": 0,
                        "queries": set(),
                        "reasons": [],
                    },
                )

                candidate["base_score"] += result.score

                candidate["queries"].add(response.query)

                candidate["reasons"].append(f"Exact Search 命中: " f"{response.query}")

        ranked_files: list[RelevantFile] = []

        for path, candidate in candidates.items():
            file_record = file_map.get(path)

            if file_record is None:
                continue

            score = candidate["base_score"]

            reasons = list(candidate["reasons"])

            queries = sorted(candidate["queries"])

            # 多 Query 命中是一个较强信号
            if len(queries) > 1:
                multi_query_bonus = (len(queries) - 1) * 20

                score += multi_query_bonus

                reasons.append(f"被 {len(queries)} 个不同 Query 命中")
            # 根据 Task Type 给不同文件类型加权
            category_bonus = self.CATEGORY_BONUS.get(
                task.task_type,
                {},
            ).get(
                file_record.category,
                0,
            )

            if category_bonus:
                score += category_bonus

                reasons.append(
                    f"{task.task_type} Task "
                    f"优先考虑 "
                    f"{file_record.category} 文件"
                )
            # target_modules 作为路径相关性信号
            normalized_path = path.lower()

            for target in task.target_modules:
                normalized_target = target.strip().lower().replace("\\", "/")
                if not normalized_target:
                    continue

                if normalized_target in normalized_path:
                    score += 25

                    reasons.append("路径匹配目标模块: " f"{target}")

            ranked_files.append(
                RelevantFile(
                    relative_path=path,
                    category=file_record.category,
                    language=file_record.language,
                    size_bytes=file_record.size_bytes,
                    score=score,
                    matched_queries=queries,
                    reasons=reasons,
                )
            )
        ranked_files.sort(
            key=lambda file: (
                -file.score,
                file.relative_path,
            )
        )

        selected: list[RelevantFile] = []

        skipped_by_budget: list[str] = []

        current_bytes = 0

        for file in ranked_files:

            if len(selected) >= max_files:
                skipped_by_budget.append(file.relative_path)
                continue

            if current_bytes + file.size_bytes > max_total_bytes:
                skipped_by_budget.append(file.relative_path)
                continue

            selected.append(file)

            current_bytes += file.size_bytes

        return RelevantFileSelection(
            candidate_count=len(ranked_files),
            selected_count=len(selected),
            total_selected_bytes=(current_bytes),
            selected_files=selected,
            skipped_by_budget=(skipped_by_budget),
        )
