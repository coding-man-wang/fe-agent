from agent_core.benchmark.case_model import BenchmarkCase
from agent_core.benchmark.change_surface_model import ChangeSurfaceEvaluation
from agent_core.change_surface.model import ResolvedChangeSurface


class ChangeSurfaceEvaluator:
    """评价“真正要改哪些文件”，而不是评价“搜到了多少相关文件”。"""

    def evaluate(
        self,
        *,
        case: BenchmarkCase,
        surface: ResolvedChangeSurface,
    ) -> ChangeSurfaceEvaluation:
        change_oracle = case.change_oracle()
        retrieval_oracle = case.retrieval_oracle()

        expected = set(change_oracle.expected_change_files)
        allowed = set(change_oracle.allowed_change_files)
        forbidden = set(change_oracle.forbidden_change_files)
        # Change Target 指标只评价真正的实现修改目标。
        # 测试文件属于独立角色，不应稀释 Target Precision。
        predicted = set(surface.implementation_targets)

        matched = expected & predicted
        missed = expected - predicted
        unexpected = predicted - expected - allowed
        forbidden_hits = predicted & forbidden

        target_recall = len(matched) / len(expected) if expected else 1.0
        target_precision = (
            len(matched) / len(predicted) if predicted else (1.0 if not expected else 0.0)
        )

        dependency_expected = (
            set(retrieval_oracle.required_context_files)
            | set(retrieval_oracle.useful_context_files)
        ) - expected
        dependency_matched = set(surface.dependencies) & dependency_expected
        dependency_coverage = (
            len(dependency_matched) / len(dependency_expected)
            if dependency_expected
            else 1.0
        )

        failure_reasons: list[str] = []
        if missed:
            failure_reasons.append("missed_change_target")
        if unexpected:
            failure_reasons.append("unexpected_change_target")
        if forbidden_hits:
            failure_reasons.append("forbidden_change_target")

        return ChangeSurfaceEvaluation(
            status="passed" if not failure_reasons else "failed",
            expected_change_files=sorted(expected),
            predicted_change_files=sorted(predicted),
            matched_change_files=sorted(matched),
            missed_change_files=sorted(missed),
            unexpected_change_files=sorted(unexpected),
            forbidden_change_hits=sorted(forbidden_hits),
            target_recall=target_recall,
            target_precision=target_precision,
            dependency_expected_files=sorted(dependency_expected),
            dependency_coverage=dependency_coverage,
            failure_reasons=failure_reasons,
        )
