"""D5-A: evidence-based candidate discovery without model calls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from agent_core.benchmark.context_eligibility_preflight import (
    analyze_context_eligibility,
)
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.codebase.graph.anchor_dependency_protection import (
    AnchorDependencyProtection,
)
from agent_core.codebase.graph_evidence_filter import (
    GraphEvidenceFilter,
)
from agent_core.codebase.task_context_selection import (
    TaskContextSelectionResult,
)
from agent_core.context.model import TaskContextBundle


def analyze_d5_candidates(
    *,
    selection_result: TaskContextSelectionResult,
    final_bundle: TaskContextBundle,
    surface: ResolvedChangeSurface,
    obligation_evidence_files: Sequence[str] = (),
) -> dict[str, Any]:
    """Explain possible file drops without changing Selection.

    This is advisory evidence, not permission to remove files.
    """

    def inconclusive(reason: str) -> dict[str, Any]:
        return {
            "version": "v1.4.9-P4-D5-A",
            "status": "inconclusive",
            "reason": reason,
            "candidates": [],
            "applied": False,
            "active_execution_allowed": False,
            "production_ready": False,
        }

    if final_bundle.task.task_id != surface.task_id:
        return inconclusive("task_identity_mismatch")

    context_paths = [item.relative_path for item in final_bundle.code_context]
    selection = selection_result.selection
    selected_paths = [item.relative_path for item in selection.selected_files]

    if (
        not context_paths
        or len(context_paths) != len(set(context_paths))
        or len(selected_paths) != len(set(selected_paths))
    ):
        return inconclusive("invalid_context_or_selection")

    strategy = final_bundle.verification_strategy or {}

    if not isinstance(strategy, dict):
        return inconclusive("invalid_verification_strategy")

    writable_tests = strategy.get("writable_test_files", [])

    if not isinstance(writable_tests, list) or any(
        not isinstance(p, str) for p in writable_tests
    ):
        return inconclusive("invalid_writable_test_files")

    protected = set().union(
        surface.implementation_targets,
        surface.tests,
        surface.dependencies,
        surface.references,
        obligation_evidence_files,
        writable_tests,
    )

    anchor_protected = AnchorDependencyProtection(
        anchor_count=5,
        max_depth=2,
    ).protected_paths(
        selection=selection_result.retrieval_selection,
        graph=selection_result.graph,
    )

    selected_set = set(selected_paths)
    context_set = set(context_paths)

    direct_import_protected = {
        edge.target_path
        for edge in selection_result.graph.edges
        if (edge.source_path in selected_set and edge.target_path in selected_set)
    }

    graph_proposal = GraphEvidenceFilter().propose(
        selection=deepcopy(selection),
        graph=selection_result.graph,
        protected_paths=anchor_protected,
    )

    graph_drops = set(graph_proposal.dropped_paths)

    # Graph Filter 之后才进行 Rerank。
    # 这里只检查 Semantic-only 静态可能性，
    # 不假装已经拿到了模型排名。
    graph_survivors = {
        item.relative_path for item in graph_proposal.selection.selected_files
    }

    rerank_candidate_count = len(graph_survivors)

    candidates = []
    possible_paths = []

    for item in selection.selected_files:
        path = item.relative_path
        blockers = []

        if path not in context_set:
            blockers.append("not_in_final_context")

        if item.category != "source":
            blockers.append("not_source_file")

        if path in protected:
            blockers.append("surface_or_obligation_protected")

        if path in anchor_protected:
            blockers.append("anchor_dependency_protected")

        if path in direct_import_protected:
            blockers.append("direct_import_protected")

        is_graph_candidate = path in graph_drops

        graph_evidence = any(
            reason.startswith("Graph dependency:")
            or reason.startswith("Graph dependent:")
            for reason in item.reasons
        )

        is_semantic_candidate = (
            path in graph_survivors
            and rerank_candidate_count >= 4
            and item.category == "source"
            and item.context_role == "unclassified"
            and item.target_match_type == "none"
            and not item.symbol_candidates
            and item.lexical_hit_count == 0
            and item.semantic_hit_count > 0
            and not graph_evidence
        )

        if not is_graph_candidate and not is_semantic_candidate:
            blockers.append("no_supported_filter_candidate")

        possible = not blockers

        if possible:
            possible_paths.append(path)

        candidates.append(
            {
                "path": path,
                "category": item.category,
                "context_role": item.context_role,
                "in_final_context": path in context_set,
                "surface_protected": path in protected,
                "anchor_protected": path in anchor_protected,
                "direct_import_protected": (path in direct_import_protected),
                "has_symbol_evidence": bool(item.symbol_candidates),
                "lexical_hit_count": item.lexical_hit_count,
                "semantic_hit_count": item.semantic_hit_count,
                "graph_preliminary_drop": is_graph_candidate,
                "rerank_candidate_possible": is_semantic_candidate,
                "possible": possible,
                "blockers": blockers,
            }
        )

    # Final Context 可能包含不来自 Retrieval Selection
    # 的 Obligation Evidence 文件。这些文件不能被
    # 当前 Selection Filter 当成合法删除候选。
    for item in final_bundle.code_context:
        if item.relative_path in selected_set:
            continue

        candidates.append(
            {
                "path": item.relative_path,
                "category": item.category,
                "context_role": "outside_selection",
                "in_final_context": True,
                "surface_protected": (item.relative_path in protected),
                "possible": False,
                "blockers": ["not_in_retrieval_selection"],
            }
        )

    return {
        "version": "v1.4.9-P4-D5-A",
        "task_id": surface.task_id,
        "repository_revision": selection_result.repository_revision,
        "status": (
            "candidate_possible" if possible_paths else "no_source_drop_candidate"
        ),
        "context_files": context_paths,
        "selection_files": selected_paths,
        "graph_preliminary_drops": sorted(graph_drops),
        "possible_candidate_paths": sorted(possible_paths),
        "candidates": candidates,
        "requires_shadow_validation": True,
        "requires_snapshot_validation": True,
        "applied": False,
        "active_execution_allowed": False,
        "production_ready": False,
    }
