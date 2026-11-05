from __future__ import annotations
import hashlib
from time import perf_counter

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from agent_core.benchmark.evidence_rerank_filter import (
    EvidenceProtectedRerankFilter,
)
from agent_core.benchmark.task_rerank_diagnostics import (
    TaskRerankDiagnosticRunner,
)
from agent_core.codebase.graph.anchor_dependency_protection import (
    AnchorDependencyProtection,
)
from agent_core.codebase.graph_evidence_filter import (
    GraphEvidenceFilter,
)
from agent_core.codebase.task_context_selection import (
    TaskContextSelectionResult,
    TaskContextSelectionService,
)
from agent_core.sdd.model import RequirementSpec, TaskDefinition
from agent_core.benchmark.selection_evidence_binding import (
    capture_selection_evidence,
)

logger = logging.getLogger(__name__)

ShadowMode = Literal["off", "shadow"]


@dataclass(frozen=True)
class ShadowPolicyConfig:
    mode: ShadowMode = "off"
    timeout_seconds: float = 30.0

    def __post_init__(self):
        if self.mode not in {"off", "shadow"}:
            raise ValueError("v1.4.9-P1 仅支持 off / shadow")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")


class ContextOptimizationShadowRunner:
    """
    统一执行 P3 验证过的策略组合。

    Retrieval Selection
         ↓
    Anchor Dependency Protection
         ↓
    Graph Evidence Filter
         ↓
    Task-aware Rerank Diagnostics
         ↓
    Evidence-protected Rerank Filter

    所有决策只产生诊断。
    不输出新的正式 Selection。
    """

    def __init__(
        self,
        *,
        rerank_runner: TaskRerankDiagnosticRunner,
    ):
        self.rerank_runner = rerank_runner

    async def observe(
        self,
        *,
        repository_root: str | Path,
        raw_requirement: str,
        result: TaskContextSelectionResult,
    ) -> dict[str, Any]:

        # 保护集合只从当前任务真实证据构建。
        # 不能使用 Benchmark Oracle。
        protected_paths = AnchorDependencyProtection(
            anchor_count=5,
            max_depth=2,
        ).protected_paths(
            selection=result.retrieval_selection,
            graph=result.graph,
        )

        # 必须使用本地 P3-D4 已验证的接口。
        graph_proposal = GraphEvidenceFilter().propose(
            selection=result.selection,
            graph=result.graph,
            protected_paths=protected_paths,
        )

        diagnostic = await self.rerank_runner.run(
            query=raw_requirement,
            repository_root=repository_root,
            selection=graph_proposal.selection,
        )

        # Rerank 不完整时，整体放弃过滤建议。
        if diagnostic.status != "complete":
            return {
                "status": "abstained",
                "reason": "rerank_diagnostic_incomplete",
                "rerank_status": diagnostic.status,
                "rerank_issues": diagnostic.issues,
                "graph_proposed_drops": (graph_proposal.dropped_paths),
                "rerank_proposed_drops": [],
                "effective_proposed_drops": [],
            }

        rerank_proposal = EvidenceProtectedRerankFilter().propose(
            selection=graph_proposal.selection,
            diagnostic=diagnostic,
            protected_paths=protected_paths,
        )

        if rerank_proposal.status == "abstained":
            return {
                "status": "abstained",
                "reason": "rerank_filter_abstained",
                "rerank_status": diagnostic.status,
                "rerank_issues": diagnostic.issues,
                "graph_proposed_drops": (graph_proposal.dropped_paths),
                "rerank_proposed_drops": [],
                "effective_proposed_drops": [],
            }

        before = {item.relative_path for item in result.selection.selected_files}

        after = {
            item.relative_path for item in rerank_proposal.selection.selected_files
        }

        return {
            "status": "complete",
            "baseline_file_count": len(before),
            "proposed_file_count": len(after),
            "graph_proposed_drops": (graph_proposal.dropped_paths),
            "rerank_proposed_drops": (rerank_proposal.proposed_drops),
            "effective_proposed_drops": sorted(before - after),
            "protected_paths": sorted(protected_paths & before),
            "rerank_status": diagnostic.status,
            "rerank_filter_status": (rerank_proposal.status),
        }


class ShadowTaskContextSelectionService:
    """
    包装原 TaskContextSelectionService。

    核心 Contract：
    - off：只运行原服务
    - shadow：传入深拷贝进行观察
    - 观察失败：记录诊断，不影响原结果
    - 永远返回 delegate.select() 的原结果
    """

    def __init__(
        self,
        *,
        delegate: TaskContextSelectionService,
        config: ShadowPolicyConfig,
        runner: ContextOptimizationShadowRunner | None = None,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ):
        if config.mode == "shadow" and runner is None:
            raise ValueError("shadow 模式必须提供 runner")

        self.delegate = delegate
        self.config = config
        self.runner = runner
        self.emit = emit

    def _emit(self, report: dict[str, Any]):
        if self.emit is None:
            return

        try:
            self.emit(report)
        except Exception:
            # 诊断输出失败，不影响正式 Selection。
            logger.exception("Shadow report sink failed")

    async def select(
        self,
        *,
        repository_root: str | Path,
        repository_revision: str,
        task: TaskDefinition,
        raw_requirement: str,
        requirement_spec: RequirementSpec,
    ) -> TaskContextSelectionResult:

        # 正式检索只执行一次。
        result = await self.delegate.select(
            repository_root=repository_root,
            repository_revision=repository_revision,
            task=task,
            raw_requirement=raw_requirement,
            requirement_spec=requirement_spec,
        )

        if self.config.mode == "off":
            return result

        baseline = result.selection.model_dump(mode="json")

        report: dict[str, Any] = {
            "policy_version": "v1.4.9-P0",
            "mode": "shadow",
            "task_id": task.task_id,
            "repository_revision": result.repository_revision,
            "applied": False,
        }

        started_at = perf_counter()

        report["requirement_sha256"] = hashlib.sha256(
            raw_requirement.encode("utf-8")
        ).hexdigest()

        try:
            assert self.runner is not None

            # 1. Shadow 前的证据状态
            before = capture_selection_evidence(
                repository_root=repository_root,
                selection=result.selection,
            )

            # 2. 观察器只接收深拷贝
            observation = await asyncio.wait_for(
                self.runner.observe(
                    repository_root=repository_root,
                    raw_requirement=raw_requirement,
                    result=deepcopy(result),
                ),
                timeout=self.config.timeout_seconds,
            )

            # 3. Shadow 后重新读取真实源码
            after = capture_selection_evidence(
                repository_root=repository_root,
                selection=result.selection,
            )

            source_matched = before.source_sha256 == after.source_sha256

            selection_matched = before.selection_sha256 == after.selection_sha256

            changed_paths = sorted(
                path
                for path in (set(before.file_sha256) | set(after.file_sha256))
                if (before.file_sha256.get(path) != after.file_sha256.get(path))
            )

            report["evidence_binding"] = {
                "selection_sha256": before.selection_sha256,
                "source_sha256_before": before.source_sha256,
                "source_sha256_after": after.source_sha256,
                "source_matched": source_matched,
                "selection_matched": selection_matched,
                "changed_paths": changed_paths,
            }

            if source_matched and selection_matched:
                # 两次证据检查一致，接受观察报告。
                report.update(observation)
            else:
                # 证据已变化，旧排名不能继续作为有效建议。
                report.update(
                    {
                        "status": "abstained",
                        "reason": "evidence_changed_during_shadow",
                        "graph_proposed_drops": [],
                        "rerank_proposed_drops": [],
                        "effective_proposed_drops": [],
                    }
                )

        except TimeoutError:
            report.update(
                {
                    "status": "error",
                    "reason": "shadow_timeout",
                }
            )

        except Exception as exc:
            report.update(
                {
                    "status": "error",
                    "reason": (f"shadow_exception:" f"{type(exc).__name__}"),
                }
            )
        finally:
            report["elapsed_ms"] = round(
                (perf_counter() - started_at) * 1000,
                2,
            )

        report["selection_unchanged"] = (
            result.selection.model_dump(mode="json") == baseline
        )

        self._emit(report)

        # 最重要的安全边界：
        # 不返回 Graph/Rerank Proposal。
        return result
