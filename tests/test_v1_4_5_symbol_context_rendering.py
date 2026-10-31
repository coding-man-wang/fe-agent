"""完整验证符号证据从文件选择到模型上下文的只读传递。"""

from __future__ import annotations

import json

import pytest

from agent_core.codebase.model import (
    CodeChunk,
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryProfile,
)
from agent_core.codebase.symbol_retrieval_model import SymbolCandidate
from agent_core.context.builder import ProjectContextBuilder
from agent_core.context.renderer import ContextRenderer
from agent_core.sdd.model import RequirementSpec, TaskDefinition, TechnicalSpec


def make_candidates():
    return [
        SymbolCandidate(
            symbol_id="orders.ts:1:loadOrders",
            name="loadOrders",
            kind="function",
            relative_path="orders.ts",
            start_line=1,
            end_line=3,
            score=1.0,
            source="name",
            matched_queries=["loadOrders"],
            reasons=["符号名称精确匹配"],
        ),
        SymbolCandidate(
            symbol_id="orders.ts:5:normalize",
            name="normalize",
            kind="function",
            relative_path="orders.ts",
            start_line=5,
            end_line=7,
            score=0.5,
            source="graph",
            depth=1,
            seed_symbol_id="orders.ts:1:loadOrders",
            relation="calls",
            direction="outgoing",
            matched_queries=["loadOrders"],
            reasons=["loadOrders 调用 normalize"],
        ),
    ]


def build_bundle(root, mode, *, with_symbols=True):
    content = (
        "export function loadOrders() {\n  return []\n}\n"
        "\nfunction normalize() {\n  return true\n}\n"
    )
    if mode != "full_file":
        content += "// filler\n" * 50
    (root / "orders.ts").write_text(content, encoding="utf-8")
    chunk = None
    if mode == "semantic_chunk":
        chunk = CodeChunk(
            chunk_id="orders.ts:1-3",
            relative_path="orders.ts",
            start_line=1,
            end_line=3,
            content="export function loadOrders() {}",
            language="TypeScript",
        )
    selected_file = HybridRelevantFile(
        relative_path="orders.ts",
        category="source",
        language="TypeScript",
        size_bytes=len(content.encode("utf-8")),
        retrieval_score=0.03,
        retrieval_rank=1,
        policy_score=0,
        selection_score=0.02,
        context_role="reference",
        best_semantic_chunk=chunk,
        reasons=["原文件检索证据"],
        symbol_candidates=make_candidates() if with_symbols else [],
    )
    selection = HybridRelevantFileSelection(
        candidate_count=1,
        eligible_count=1,
        selected_count=1,
        total_selected_bytes=selected_file.size_bytes,
        selected_files=[selected_file],
    )
    bundle = ProjectContextBuilder().build(
        repository_root=root,
        requirement=RequirementSpec(title="订单查询", summary="检查订单查询逻辑"),
        technical_spec=TechnicalSpec(summary="沿现有查询函数阅读代码"),
        task=TaskDefinition(
            task_id="TASK-SYMBOL-CONTEXT",
            task_type="code_change",
            title="检查订单查询",
            objective="检查 loadOrders",
            source_step_order=1,
        ),
        repository=RepositoryProfile(project_name="fixture", root_path=str(root)),
        selection=selection,
        max_files=1,
        max_total_chars=1000 if mode == "full_file" else 40,
        max_chars_per_file=1000 if mode == "full_file" else 40,
    )
    return bundle, selected_file


@pytest.mark.parametrize("mode", ["full_file", "semantic_chunk", "truncated"])
def test_symbol_evidence_reaches_rendered_context_in_every_content_mode(tmp_path, mode):
    bundle, selected = build_bundle(tmp_path, mode)
    item = bundle.code_context[0]
    assert item.content_mode == mode
    assert item.symbol_candidates == selected.symbol_candidates
    assert item.relevance_reasons == ["原文件检索证据"]
    assert item.content_chars == len(item.content)

    payload = json.loads(ContextRenderer().render(bundle).user_message)
    rendered_item = payload["code_context"][0]
    evidence = rendered_item["symbol_evidence"]
    assert evidence["purpose"] == "read_context_only"
    assert "不授予文件修改权限" in evidence["permission_note"]
    assert evidence["candidates"] == [
        candidate.model_dump(mode="json") for candidate in selected.symbol_candidates
    ]
    graph_candidate = evidence["candidates"][1]
    assert graph_candidate["seed_symbol_id"] == "orders.ts:1:loadOrders"
    assert graph_candidate["relation"] == "calls"
    assert graph_candidate["direction"] == "outgoing"
    assert graph_candidate["depth"] == 1
    assert payload["change_surface"] is None
    assert selected.target_match_type == "none"
    assert selected.context_role == "reference"


@pytest.mark.parametrize("mode", ["full_file", "semantic_chunk", "truncated"])
def test_symbol_metadata_does_not_replace_source_or_change_existing_budget(
    tmp_path, mode
):
    original, _ = build_bundle(tmp_path, mode, with_symbols=False)
    enhanced, _ = build_bundle(tmp_path, mode, with_symbols=True)
    renderer = ContextRenderer()
    before = renderer.render(original)
    after = renderer.render(enhanced)
    before_payload = json.loads(before.user_message)
    after_payload = json.loads(after.user_message)
    assert before.system_message == after.system_message
    assert enhanced.budget == original.budget
    assert "symbol_evidence" not in before_payload["code_context"][0]
    after_payload["code_context"][0].pop("symbol_evidence")
    assert after_payload == before_payload


def test_copying_evidence_does_not_mutate_the_retrieval_candidate(tmp_path):
    bundle, selected = build_bundle(tmp_path, "full_file")
    bundle.code_context[0].symbol_candidates[0].reasons.append("上下文层附加说明")
    assert selected.symbol_candidates[0].reasons == ["符号名称精确匹配"]


def test_existing_write_scope_is_not_extended_by_symbol_evidence(tmp_path):
    bundle, _ = build_bundle(tmp_path, "full_file")
    scope = {
        "implementation_targets": ["another.ts"],
        "tests": [],
        "dependencies": [],
        "references": ["orders.ts"],
        "create_files": [],
    }
    bundle.change_surface = scope
    payload = json.loads(ContextRenderer().render(bundle).user_message)
    assert payload["change_surface"] == scope
    assert "orders.ts" not in payload["change_surface"]["implementation_targets"]
    assert payload["code_context"][0]["symbol_evidence"]["candidates"]
