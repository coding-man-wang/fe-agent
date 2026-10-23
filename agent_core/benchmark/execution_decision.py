from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from agent_core.change_surface.model import (
    ResolvedChangeSurface,
)
from agent_core.coding.model import (
    EvidenceGap,
    ReadOnlyCodingResult,
)

RuntimeExecutionAction = Literal[
    "continue",
    "skip_runtime_noop",
    "block_insufficient_evidence",
]


class RuntimeExecutionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: RuntimeExecutionAction

    reason: str


class RuntimeExecutionDecisionPolicy:
    """
    Fresh Commit Analysis
            ↓
    Runtime Execution Decision

    模型负责表达：
        blocking / advisory

    Harness 负责：
        deterministic safety escalation
    """

    def decide(
        self,
        result: ReadOnlyCodingResult,
        *,
        surface: ResolvedChangeSurface,
    ) -> RuntimeExecutionDecision:

        # -----------------------------------
        # 1. 已经存在修改目标
        # -----------------------------------

        if result.proposed_changes:
            return RuntimeExecutionDecision(
                action="continue",
                reason=(
                    f"coding analysis proposed "
                    f"{len(result.proposed_changes)} "
                    "change target(s)"
                ),
            )

        # -----------------------------------
        # 2. 新结构化 Evidence Gap
        # -----------------------------------

        if result.evidence_gaps:
            return self._decide_from_evidence_gaps(
                gaps=result.evidence_gaps,
                surface=surface,
            )

        # -----------------------------------
        # 3. Legacy fallback
        #
        # 老模型只有 missing_information。
        # 无法判断 severity，因此继续保守阻断。
        # -----------------------------------

        legacy_missing = [
            item.strip() for item in result.missing_information if item and item.strip()
        ]

        if legacy_missing:
            return RuntimeExecutionDecision(
                action="block_insufficient_evidence",
                reason=(
                    "legacy unstructured missing information "
                    "cannot be proven advisory: " + " | ".join(legacy_missing)
                ),
            )

        # -----------------------------------
        # 4. 无修改 + 无 Evidence Gap
        # -----------------------------------

        return RuntimeExecutionDecision(
            action="skip_runtime_noop",
            reason=(
                "fresh-commit coding analysis produced "
                "no changes and no blocking evidence gaps"
            ),
        )

    def _decide_from_evidence_gaps(
        self,
        *,
        gaps: list[EvidenceGap],
        surface: ResolvedChangeSurface,
    ) -> RuntimeExecutionDecision:

        blocking: list[EvidenceGap] = []

        advisory: list[EvidenceGap] = []

        for gap in gaps:

            # -----------------------------------
            # 1. Verification-stage Evidence
            #
            # 这些信息不应该阻止 Coding Decision。
            #
            # 例如：
            # - 测试尚未执行
            # - runtime log 尚未产生
            # - 请求轨迹尚未产生
            # - UI interaction 尚未动态验证
            #
            # 它们应该交给后续 Verification Gate。
            # -----------------------------------

            if gap.resolution_stage == "verification":
                advisory.append(gap)
                continue

            # -----------------------------------
            # 2. Analysis-stage Blocking Gap
            # -----------------------------------

            if gap.severity == "blocking":
                blocking.append(gap)
                continue

            # -----------------------------------
            # 3. Harness Safety Escalation
            #
            # 只有 analysis-stage gap 才执行
            # deterministic safety escalation。
            #
            # 即使模型声称 advisory，
            # 如果它实际上指向 implementation target，
            # Harness 仍保守阻断。
            # -----------------------------------

            if self._touches_implementation_target(
                gap=gap,
                surface=surface,
            ):
                blocking.append(gap)
                continue

            # -----------------------------------
            # 4. 真正 Advisory
            # -----------------------------------

            advisory.append(gap)

        if blocking:
            return RuntimeExecutionDecision(
                action="block_insufficient_evidence",
                reason=(
                    "blocking evidence gap(s): "
                    + " | ".join(gap.description for gap in blocking)
                ),
            )

        return RuntimeExecutionDecision(
            action="skip_runtime_noop",
            reason=(
                "fresh-commit coding analysis produced "
                "no proposed changes; all remaining "
                "evidence gaps are advisory"
                + (
                    ": " + " | ".join(gap.description for gap in advisory)
                    if advisory
                    else ""
                )
            ),
        )

    @staticmethod
    def _touches_implementation_target(
        *,
        gap: EvidenceGap,
        surface: ResolvedChangeSurface,
    ) -> bool:

        targets = {
            path.replace("\\", "/").lower() for path in surface.implementation_targets
        }

        target_names = {path.rsplit("/", 1)[-1] for path in targets}

        # -----------------------------------
        # 首先检查结构化 related_paths
        # -----------------------------------

        for path in gap.related_paths:

            normalized = path.replace("\\", "/").strip().lower()

            if normalized in targets:
                return True

        # -----------------------------------
        # Defensive fallback：
        #
        # 防止模型 related_paths 漏填，
        # 但 description 明确提到了 target。
        # -----------------------------------

        description = gap.description.replace("\\", "/").lower()

        for target in targets:
            if target in description:
                return True

        for name in target_names:
            if name and name in description:
                return True

        return False
