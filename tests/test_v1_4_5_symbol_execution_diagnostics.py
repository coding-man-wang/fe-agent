"""符号阅读诊断进入执行记录，同时保持现有执行与权限数据。"""

from types import SimpleNamespace
from unittest.mock import Mock

from agent_core.benchmark.execution_model import FullExecutionTaskRecord
from agent_core.benchmark.full_execution import _context_selection_diagnostics


def selection(paths):
    return SimpleNamespace(
        selected_files=[SimpleNamespace(relative_path=path) for path in paths]
    )


def context(**overrides):
    data = {
        "retrieval_selection": selection(["src/base.ts"]),
        "selection": selection(["src/base.ts", "src/dependency.ts", "src/symbol.ts"]),
        "retrieval_mode": "hybrid",
        "semantic_error": None,
        "symbol_added_files": ["src/symbol.ts"],
        "symbol_error": None,
        "symbol_issues": [],
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def record(**overrides):
    data = {
        "task_id": "TASK-SYMBOL",
        "task_type": "code_change",
        "status": "completed",
        "proposed_files": ["src/base.ts"],
        "applied_policy_rule_ids": ["policy-original"],
        "scheduled_verification_command_ids": ["typecheck"],
    }
    data.update(overrides)
    return FullExecutionTaskRecord(**data)


def test_symbol_and_file_graph_additions_are_reported_separately():
    original = record()
    updated = original.model_copy(
        deep=True,
        update=_context_selection_diagnostics(context()),
    )
    assert updated.retrieval_files == ["src/base.ts"]
    assert updated.selected_context_files == [
        "src/base.ts",
        "src/dependency.ts",
        "src/symbol.ts",
    ]
    assert updated.graph_added_files == ["src/dependency.ts"]
    assert updated.symbol_added_files == ["src/symbol.ts"]
    assert updated.proposed_files == original.proposed_files
    assert updated.applied_policy_rule_ids == original.applied_policy_rule_ids
    assert updated.scheduled_verification_command_ids == ["typecheck"]
    assert updated.planned_impact is original.planned_impact is None
    assert updated.planned_verification is original.planned_verification is None
    assert original.selected_context_files == []


def test_diagnostics_survive_record_serialization_and_do_not_alias_source():
    result = context(
        symbol_error="RuntimeError: symbol index unavailable",
        symbol_issues=["src/broken.ts: invalid syntax"],
        symbol_added_files=[],
    )
    updated = record().model_copy(update=_context_selection_diagnostics(result))
    restored = FullExecutionTaskRecord.model_validate_json(updated.model_dump_json())
    assert restored.symbol_fallback_reason == "RuntimeError: symbol index unavailable"
    assert restored.symbol_issues == ["src/broken.ts: invalid syntax"]
    assert restored.symbol_added_files == []
    result.symbol_issues.append("late issue")
    assert updated.symbol_issues == restored.symbol_issues


def test_legacy_context_and_legacy_record_keep_empty_symbol_diagnostics():
    legacy = SimpleNamespace(
        retrieval_selection=selection(["src/base.ts"]),
        selection=selection(["src/base.ts", "src/dependency.ts"]),
        retrieval_mode="lexical_only",
        semantic_error=None,
    )
    original = FullExecutionTaskRecord.model_validate(
        {
            "task_id": "TASK-OLD",
            "task_type": "code_change",
            "status": "completed",
        }
    )
    assert original.symbol_added_files == original.symbol_issues == []
    assert original.symbol_fallback_reason is None
    updated = original.model_copy(update=_context_selection_diagnostics(legacy))
    assert updated.graph_added_files == ["src/dependency.ts"]
    assert updated.symbol_added_files == updated.symbol_issues == []
    assert updated.symbol_fallback_reason is None


def test_unconfigured_legacy_mock_does_not_create_fake_symbol_diagnostics():
    legacy = Mock()
    legacy.retrieval_selection = selection(["src/base.ts"])
    legacy.selection = selection(["src/base.ts", "src/dependency.ts"])
    legacy.retrieval_mode = "lexical_only"
    legacy.semantic_error = None
    updated = record().model_copy(update=_context_selection_diagnostics(legacy))
    assert updated.graph_added_files == ["src/dependency.ts"]
    assert updated.symbol_added_files == updated.symbol_issues == []
    assert updated.symbol_fallback_reason is None


def test_symbol_restoration_of_retrieval_file_retains_symbol_provenance():
    # 文件图可能已移除一个检索文件，符号阶段又在剩余预算中找回它。
    result = context(
        retrieval_selection=selection(["src/base.ts", "src/restored.ts"]),
        selection=selection(["src/base.ts", "src/dependency.ts", "src/restored.ts"]),
        symbol_added_files=["src/restored.ts"],
    )
    diagnostics = _context_selection_diagnostics(result)
    assert diagnostics["symbol_added_files"] == ["src/restored.ts"]
    assert diagnostics["graph_added_files"] == ["src/dependency.ts"]


def test_semantic_and_symbol_fallbacks_remain_independent():
    result = context(
        retrieval_mode="lexical_fallback",
        semantic_error="embedding unavailable",
        symbol_error="symbol parser unavailable",
        symbol_added_files=[],
    )
    updated = record().model_copy(update=_context_selection_diagnostics(result))
    assert updated.context_retrieval_mode == "lexical_fallback"
    assert updated.semantic_fallback_reason == "embedding unavailable"
    assert updated.symbol_fallback_reason == "symbol parser unavailable"
