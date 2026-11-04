from pathlib import Path

from agent_core.benchmark.context_comparison_model import (
    ContextComparisonResult,
    ContextQualityMetrics,
)
from agent_core.benchmark.symbol_window_evaluator import (
    SymbolWindowBenchmarkCase,
    SymbolWindowBenchmarkEvaluator,
)
from agent_core.code_intelligence.symbol_index import SymbolIndex
from agent_core.context.model import TaskContextBundle


class ContextComparisonEvaluator:
    def __init__(self) -> None:
        self.evaluator = SymbolWindowBenchmarkEvaluator()

    def evaluate(
        self,
        *,
        case_id: str,
        repository_root: str | Path,
        symbol_index: SymbolIndex,
        required_symbol_ids: list[str],
        baseline_bundle: TaskContextBundle,
        enhanced_bundle: TaskContextBundle,
        irrelevant_paths: set[str] | None = None,
    ) -> ContextComparisonResult:

        required = set(required_symbol_ids)
        if not required:
            raise ValueError("必须提供独立定义的 Required Symbols")

        baseline_paths = {item.relative_path for item in baseline_bundle.code_context}
        enhanced_paths = {item.relative_path for item in enhanced_bundle.code_context}

        if baseline_paths != enhanced_paths:
            raise ValueError("A/B File Selection 不一致，无法隔离压缩策略的影响")

        noise_paths = irrelevant_paths or set()
        if noise_paths - baseline_paths:
            raise ValueError("Noise Oracle 包含未选择的文件")

        case = SymbolWindowBenchmarkCase(
            case_id=case_id,
            required_symbol_ids=sorted(required),
            require_symbol_window=False,
            require_independent_symbol_index=True,
        )

        baseline_eval = self.evaluator.evaluate(
            case=case,
            bundle=baseline_bundle,
            repository_root=repository_root,
            symbol_index=symbol_index,
        )
        enhanced_eval = self.evaluator.evaluate(
            case=case,
            bundle=enhanced_bundle,
            repository_root=repository_root,
            symbol_index=symbol_index,
        )

        baseline = self._metrics(baseline_bundle, baseline_eval, noise_paths)
        enhanced = self._metrics(enhanced_bundle, enhanced_eval, noise_paths)

        saved_chars = baseline.context_chars - enhanced.context_chars
        reduction_ratio = (
            saved_chars / baseline.context_chars if baseline.context_chars else 0.0
        )
        coverage_delta = enhanced.symbol_coverage - baseline.symbol_coverage

        no_regression = coverage_delta >= -1e-9

        # Baseline 允许缺失必要符号，但不得伪造源码或元数据
        baseline_integrity = (
            baseline.budget_preserved
            and baseline.source_integrity_preserved
            and baseline.metadata_integrity_preserved
        )
        enhanced_integrity = (
            enhanced.budget_preserved
            and enhanced.source_integrity_preserved
            and enhanced.metadata_integrity_preserved
        )

        issues = []

        if not baseline_integrity:
            issues.append("baseline_integrity_failed")
        if not enhanced_integrity:
            issues.append("enhanced_integrity_failed")
        if not no_regression:
            issues.append("symbol_coverage_regression")
        if enhanced.symbol_coverage < 1.0:
            issues.append("enhanced_required_symbol_missing")

        return ContextComparisonResult(
            case_id=case_id,
            status="passed" if not issues else "failed",
            baseline=baseline,
            enhanced=enhanced,
            saved_chars=saved_chars,
            reduction_ratio=reduction_ratio,
            coverage_delta=coverage_delta,
            no_coverage_regression=no_regression,
            issues=issues,
        )

    @staticmethod
    def _metrics(
        bundle,
        evaluation,
        noise_paths: set[str],
    ) -> ContextQualityMetrics:
        total_chars = sum(len(item.content) for item in bundle.code_context)
        noise_chars = sum(
            len(item.content)
            for item in bundle.code_context
            if item.relative_path in noise_paths
        )

        return ContextQualityMetrics(
            used_files=len(bundle.code_context),
            context_chars=total_chars,
            required_symbol_count=len(evaluation.required_symbol_ids),
            covered_symbol_count=(
                len(evaluation.covered_symbol_ids)
                + len(evaluation.localized_symbol_ids)
            ),
            symbol_coverage=evaluation.symbol_coverage,
            missing_symbol_ids=evaluation.missing_symbol_ids,
            budget_preserved=evaluation.budget_preserved,
            source_integrity_preserved=(evaluation.source_integrity_preserved),
            metadata_integrity_preserved=(evaluation.metadata_integrity_preserved),
            noise_chars=noise_chars,
            noise_ratio=(noise_chars / total_chars if total_chars else 0.0),
        )
