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
    """
    Benchmark Retrieval 自动阅卷器。

    v0.1 只评价核心代码 Retrieval。

    默认参与评分的 Task 类型：

        investigation
        code_change

    test / verification 暂时不进入
    核心代码 Retrieval Precision / Recall。
    """

    DEFAULT_SCORED_TASK_TYPES = {
        "investigation",
        "code_change",
    }

    def __init__(
        self,
        *,
        scored_task_types: set[str] | None = None,
    ) -> None:

        self.scored_task_types = (
            set(scored_task_types)
            if scored_task_types is not None
            else set(self.DEFAULT_SCORED_TASK_TYPES)
        )

        if not self.scored_task_types:
            raise ValueError("scored_task_types 不能为空")

    def evaluate(
        self,
        *,
        case: BenchmarkCase,
        capture: BenchmarkRetrievalCapture,
    ) -> BenchmarkRetrievalEvaluation:

        # ===================================
        # 1. Hidden Oracle
        # ===================================

        expected_files = self._normalize_paths(case.expectation.expected_files)

        forbidden_paths = self._normalize_paths(case.expectation.forbidden_files)

        # ===================================
        # 2. 收集参与评分的 Retrieval
        # ===================================

        retrieved: set[str] = set()

        for task_capture in capture.task_retrievals:

            if task_capture.task_type not in self.scored_task_types:
                continue

            for selected_file in task_capture.selected_files:

                retrieved.add(self._normalize_path(selected_file.relative_path))

        retrieved_files = sorted(retrieved)

        expected_set = set(expected_files)

        retrieved_set = set(retrieved_files)

        # ===================================
        # 3. Recall
        # ===================================

        matched = sorted(expected_set & retrieved_set)

        missed = sorted(expected_set - retrieved_set)

        if expected_set:

            recall = len(matched) / len(expected_set)

        else:
            # 当前 Case Schema 要求行为标准，
            # 但 expected_files 可以为空。
            #
            # 没有目标文件时，
            # 文件 Recall 不适用。
            #
            # v0.1 记为 1，表示不存在漏检。
            recall = 1.0

        # ===================================
        # 4. Precision
        # ===================================

        unexpected = sorted(retrieved_set - expected_set)

        if retrieved_set:

            precision = len(matched) / len(retrieved_set)

        else:
            # 如果预期存在文件，
            # 但一个都没找到，
            # Precision 定义为 0。
            #
            # 如果本来就没有 expected file，
            # 则没有产生噪声，记为 1。
            precision = 0.0 if expected_set else 1.0

        # ===================================
        # 5. Forbidden Path Check
        # ===================================

        forbidden_hits = sorted(
            path
            for path in retrieved_set
            if self._matches_forbidden(
                path=path,
                forbidden_paths=(forbidden_paths),
            )
        )

        # ===================================
        # 6. Failure Taxonomy
        # ===================================

        failure_reasons: list[str] = []

        if missed:

            failure_reasons.append("retrieval_miss")

        if unexpected:

            failure_reasons.append("retrieval_noise")

        if forbidden_hits:

            failure_reasons.append("forbidden_scope_retrieval")

        # ===================================
        # 7. Overall Status
        # ===================================

        if recall == 1.0 and not forbidden_hits:

            status = "passed"

        elif recall > 0.0:

            status = "partial"

        else:

            status = "failed"

        return BenchmarkRetrievalEvaluation(
            status=status,
            scored_task_types=sorted(self.scored_task_types),
            expected_files=(expected_files),
            retrieved_files=(retrieved_files),
            matched_expected_files=(matched),
            missed_expected_files=(missed),
            unexpected_files=(unexpected),
            forbidden_hits=(forbidden_hits),
            recall=recall,
            precision=precision,
            failure_reasons=(failure_reasons),
        )

    @staticmethod
    def _normalize_paths(
        paths: list[str],
    ) -> list[str]:

        return sorted(
            {
                BenchmarkRetrievalEvaluator._normalize_path(path)
                for path in paths
                if path.strip()
            }
        )

    @staticmethod
    def _normalize_path(
        path: str,
    ) -> str:

        normalized = path.strip().replace("\\", "/")

        while normalized.startswith("./"):
            normalized = normalized[2:]

        return normalized.rstrip("/")

    @staticmethod
    def _matches_forbidden(
        *,
        path: str,
        forbidden_paths: list[str],
    ) -> bool:

        normalized_path = path.rstrip("/")

        for forbidden in forbidden_paths:

            normalized_forbidden = forbidden.rstrip("/")

            if not normalized_forbidden:
                continue

            if normalized_path == normalized_forbidden:
                return True

            if normalized_path.startswith(normalized_forbidden + "/"):
                return True

        return False
