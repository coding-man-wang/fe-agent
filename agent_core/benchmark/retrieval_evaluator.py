from agent_core.benchmark.case_model import (
    BenchmarkCase,
)
from agent_core.benchmark.evaluation_model import (
    BenchmarkRetrievalEvaluation,
)
from agent_core.benchmark.retrieval_model import (
    BenchmarkRetrievalCapture,
)


class BenchmarkRetrievalEvaluator:

    DEFAULT_SCORED_TASK_TYPES = {
        "investigation",
        "code_change",
    }

    def __init__(
        self,
        *,
        scored_task_types: set[str] | None = None,
    ) -> None:

        self.scored_task_types = scored_task_types or self.DEFAULT_SCORED_TASK_TYPES

    def evaluate(
        self,
        *,
        case: BenchmarkCase,
        capture: BenchmarkRetrievalCapture,
    ) -> BenchmarkRetrievalEvaluation:

        oracle = case.retrieval_oracle()

        required = {
            self._normalize_path(path) for path in (oracle.required_context_files)
        }

        useful = {self._normalize_path(path) for path in (oracle.useful_context_files)}

        forbidden = {
            self._normalize_path(path) for path in (oracle.forbidden_context_files)
        }

        # required + useful
        # 都属于“合理 Context”。
        relevant = required | useful

        retrieved: set[str] = set()

        for task_retrieval in capture.task_retrievals:

            if task_retrieval.task_type not in self.scored_task_types:
                continue

            for selected_file in task_retrieval.selected_files:

                retrieved.add(self._normalize_path(selected_file.relative_path))

        matched_required = required & retrieved

        missed_required = required - retrieved

        # Precision 不再只认 required。
        #
        # useful context 被搜到也是好事。
        matched_relevant = relevant & retrieved

        unexpected = retrieved - relevant

        forbidden_hits = {
            path
            for path in retrieved
            if self._matches_any_scope(
                path,
                forbidden,
            )
        }

        # ---------------------------------
        # Recall
        #
        # 必须找到的文件，
        # 到底找回来了多少。
        # ---------------------------------

        if required:

            recall = len(matched_required) / len(required)

        else:

            recall = 1.0

        # ---------------------------------
        # Precision
        #
        # Agent 找回来的一堆文件中，
        # 有多少是真正有用的。
        # ---------------------------------

        if retrieved:

            precision = len(matched_relevant) / len(retrieved)

        else:

            precision = 1.0 if not relevant else 0.0

        failure_reasons: list[str] = []

        if missed_required:

            failure_reasons.append("retrieval_miss")

        if unexpected:

            failure_reasons.append("retrieval_noise")

        if forbidden_hits:

            failure_reasons.append("forbidden_scope_retrieval")

        if recall == 1.0 and not forbidden_hits:

            status = "passed"

        elif recall > 0.0:

            status = "partial"

        else:

            status = "failed"

        return BenchmarkRetrievalEvaluation(
            status=status,
            scored_task_types=sorted(self.scored_task_types),
            # evaluation_model 暂时继续沿用
            # v1 字段名，
            # 避免这一步改动太大。
            #
            # 这里的 expected_files
            # 从现在起表示：
            # required context files。
            expected_files=sorted(required),
            retrieved_files=sorted(retrieved),
            matched_expected_files=sorted(matched_required),
            missed_expected_files=sorted(missed_required),
            unexpected_files=sorted(unexpected),
            forbidden_hits=sorted(forbidden_hits),
            recall=recall,
            precision=precision,
            failure_reasons=(failure_reasons),
        )

    @staticmethod
    def _normalize_path(
        value: str,
    ) -> str:

        return value.replace(
            "\\",
            "/",
        ).strip("/")

    @classmethod
    def _matches_any_scope(
        cls,
        path: str,
        scopes: set[str],
    ) -> bool:

        normalized_path = cls._normalize_path(path)

        for scope in scopes:

            normalized_scope = cls._normalize_path(scope)

            if normalized_path == normalized_scope:
                return True

            if normalized_path.startswith(normalized_scope + "/"):
                return True

        return False
