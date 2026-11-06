from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from time import perf_counter
from typing import Any

from agent_core.benchmark.selection_evidence_binding import (
    capture_selection_evidence,
    json_digest,
    sha256_bytes,
)
from agent_core.benchmark.selection_shadow_integration import (
    ContextOptimizationShadowRunner,
)
from agent_core.codebase.task_context_selection import (
    TaskContextSelectionResult,
)
from agent_core.context.model import TaskContextBundle

logger = logging.getLogger(__name__)


def _context_source_hashes(
    *,
    repository_root: str | Path,
    bundle: TaskContextBundle,
) -> dict[str, str]:
    """
    读取最终 Code Context 引用的真实文件。

    覆盖：
    - Retrieval 文件
    - Change Surface 补入文件
    - Obligation Evidence 补入文件

    不读取 Selection 以外的整个代码仓库。
    """
    root = Path(repository_root).resolve(strict=True)

    hashes: dict[str, str] = {}

    for item in bundle.code_context:
        relative = item.relative_path.replace("\\", "/")
        pure = PurePosixPath(relative)

        if (
            not pure.parts
            or pure.is_absolute()
            or PureWindowsPath(relative).drive
            or ".." in pure.parts
            or relative in hashes
        ):
            raise ValueError(f"非法或重复 Context 路径: {relative}")

        source = (root / Path(*pure.parts)).resolve(strict=True)

        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError("Context 文件超出当前仓库")

        hashes[relative] = sha256_bytes(source.read_bytes())

    return hashes


class FullExecutionContextShadowObserver:
    """
    v1.4.9-P4-B-D1

    针对已经完成 Retrieval 和 Context Build
    的 Full Execution Task 进行 Shadow 诊断。

    Contract:
    1. 不调用 selection_service.select()
    2. Graph / Rerank 只接收深拷贝
    3. 不修改实际 TaskContextBundle
    4. 只返回报告，不返回优化后的 Context
    5. 异常 / Timeout / Snapshot 变化时不接受建议
    """

    PROPOSAL_FIELDS = (
        "baseline_file_count",
        "proposed_file_count",
        "graph_proposed_drops",
        "rerank_proposed_drops",
        "effective_proposed_drops",
        "protected_paths",
        "rerank_status",
        "rerank_filter_status",
        "rerank_issues",
    )

    def __init__(
        self,
        *,
        runner: ContextOptimizationShadowRunner,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")

        self.runner = runner
        self.timeout_seconds = timeout_seconds

    async def observe(
        self,
        *,
        repository_root: str | Path,
        repository_revision: str,
        raw_requirement: str,
        selection_result: TaskContextSelectionResult,
        bundle: TaskContextBundle,
    ) -> dict[str, Any]:
        started = perf_counter()

        selection = selection_result.selection

        # 只用于诊断，不向下游返回这些副本。
        selection_before = selection.model_dump(mode="json")

        context_before = bundle.model_dump(mode="json")

        context_files_before = [item.relative_path for item in bundle.code_context]

        report: dict[str, Any] = {
            "policy_version": "v1.4.9-P4-B-D1",
            "event_type": "full_execution_context_shadow",
            "observation_point": ("before_coding_agent_messages"),
            "mode": "shadow",
            "task_id": bundle.task.task_id,
            "repository_revision": repository_revision,
            "applied": False,
            "status": "error",
            "reason": "not_observed",
            "requirement_sha256": sha256_bytes(raw_requirement.encode("utf-8")),
            "effective_proposed_drops": [],
            "context_files_before": context_files_before,
        }

        try:
            if selection_result.repository_revision != repository_revision:
                report.update(
                    {
                        "status": "abstained",
                        "reason": "repository_revision_mismatch",
                    }
                )
                return report

            if bundle.repository.root_path:
                root = Path(repository_root).resolve(strict=True)

                bundle_root = Path(bundle.repository.root_path).resolve(strict=True)

                if bundle_root != root:
                    report.update(
                        {
                            "status": "abstained",
                            "reason": "context_root_mismatch",
                        }
                    )
                    return report

            # 1. Shadow 前的 Selection 及源码证据
            selection_evidence_before = capture_selection_evidence(
                repository_root=repository_root,
                selection=selection,
            )

            # 2. 最终 Context 自身的结构与内容摘要
            bundle_digest_before = json_digest(context_before)

            # 3. 读取最终 Context 中所有文件，
            #    包括非 Retrieval 补入的证据。
            context_sources_before = _context_source_hashes(
                repository_root=repository_root,
                bundle=bundle,
            )

            # 4. 使用已有 Selection 执行 Shadow。
            #    不允许再次 Retrieval。
            observation = await asyncio.wait_for(
                self.runner.observe(
                    repository_root=repository_root,
                    raw_requirement=raw_requirement,
                    result=deepcopy(selection_result),
                ),
                timeout=self.timeout_seconds,
            )

            if not isinstance(observation, dict):
                raise TypeError("Shadow Observer 必须返回 dict")

            if observation.get("status") not in {
                "complete",
                "abstained",
            }:
                raise ValueError("Shadow 诊断状态不合法")

            # 5. Shadow 后重新读取证据。
            selection_evidence_after = capture_selection_evidence(
                repository_root=repository_root,
                selection=selection,
            )

            context_after = bundle.model_dump(mode="json")

            bundle_digest_after = json_digest(context_after)

            context_sources_after = _context_source_hashes(
                repository_root=repository_root,
                bundle=bundle,
            )

            source_matched = (
                selection_evidence_before.source_sha256
                == selection_evidence_after.source_sha256
            )

            selection_matched = (
                selection_evidence_before.selection_sha256
                == selection_evidence_after.selection_sha256
            )

            bundle_matched = bundle_digest_before == bundle_digest_after

            bundle_sources_matched = context_sources_before == context_sources_after

            all_context_paths = set(context_sources_before) | set(context_sources_after)

            changed_context_paths = sorted(
                path
                for path in all_context_paths
                if (context_sources_before.get(path) != context_sources_after.get(path))
            )

            all_selection_paths = set(selection_evidence_before.file_sha256) | set(
                selection_evidence_after.file_sha256
            )

            changed_selection_paths = sorted(
                path
                for path in all_selection_paths
                if (
                    selection_evidence_before.file_sha256.get(path)
                    != selection_evidence_after.file_sha256.get(path)
                )
            )

            report["evidence_binding"] = {
                "selection_sha256": (selection_evidence_before.selection_sha256),
                "source_sha256_before": (selection_evidence_before.source_sha256),
                "source_sha256_after": (selection_evidence_after.source_sha256),
                "source_matched": source_matched,
                "selection_matched": selection_matched,
                "changed_paths": changed_selection_paths,
            }

            report.update(
                {
                    "bundle_sha256_before": bundle_digest_before,
                    "bundle_sha256_after": bundle_digest_after,
                    "context_unchanged": bundle_matched,
                    "bundle_source_matched": (bundle_sources_matched),
                    "bundle_source_changed_paths": (changed_context_paths),
                }
            )

            valid = all(
                (
                    source_matched,
                    selection_matched,
                    bundle_matched,
                    bundle_sources_matched,
                )
            )

            if not valid:
                report.update(
                    {
                        "status": "abstained",
                        "reason": "final_context_evidence_changed",
                        "effective_proposed_drops": [],
                    }
                )

            else:
                # 仅提取已知诊断字段，避免 Runner
                # 意外覆盖 task_id、mode 或 applied。
                for key in self.PROPOSAL_FIELDS:
                    if key in observation:
                        report[key] = observation[key]

                report["status"] = observation["status"]

                if observation["status"] == "complete":
                    report.pop("reason", None)
                else:
                    report["reason"] = observation.get(
                        "reason",
                        "runner_abstained",
                    )
                    report["effective_proposed_drops"] = []

        except TimeoutError:
            report.update(
                {
                    "status": "error",
                    "reason": "shadow_timeout",
                    "effective_proposed_drops": [],
                }
            )

        except Exception as exc:
            report.update(
                {
                    "status": "error",
                    "reason": ("shadow_exception:" + type(exc).__name__),
                    "effective_proposed_drops": [],
                }
            )

            logger.exception("Full Execution Shadow Observation Failed")

        finally:
            report["selection_unchanged"] = (
                selection.model_dump(mode="json") == selection_before
            )

            report["context_unchanged"] = (
                bundle.model_dump(mode="json") == context_before
            )

            report["context_files_after"] = [
                item.relative_path for item in bundle.code_context
            ]

            report["elapsed_ms"] = round(
                (perf_counter() - started) * 1000,
                2,
            )

            # 即使 Runner 返回 applied=True，
            # 也绝不把它作为正式执行状态。
            report["applied"] = False

        return report
