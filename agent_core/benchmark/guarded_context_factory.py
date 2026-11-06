from __future__ import annotations

from pathlib import Path
from typing import Callable

from agent_core.benchmark.context_policy_runtime import (
    GuardedContextPolicyConsumer,
)
from agent_core.benchmark.selection_shadow_integration import (
    ContextOptimizationShadowRunner,
    ShadowPolicyConfig,
    ShadowTaskContextSelectionService,
)
from agent_core.codebase.model import RepositoryProfile
from agent_core.codebase.task_context_selection import (
    TaskContextSelectionService,
)
from agent_core.context.builder import ProjectContextBuilder
from agent_core.context.model import TaskContextBundle
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


class GuardedTaskContextFactory:
    """
    兼容现有 ContextFactory.build(task) 协议。

    完整链路：
      Selection → Shadow → Snapshot Gate
      → Context Policy → TaskContextBundle
    """

    def __init__(
        self,
        *,
        repository_root: str | Path,
        raw_requirement: str,
        requirement: RequirementSpec,
        technical_spec: TechnicalSpec,
        repository: RepositoryProfile,
        selection_service: TaskContextSelectionService,
        shadow_runner: ContextOptimizationShadowRunner,
        policy_consumer: GuardedContextPolicyConsumer,
        get_current_revision: Callable[[], str],
        shadow_timeout_seconds: float = 90.0,
    ):
        self.root = Path(repository_root)
        self.raw_requirement = raw_requirement
        self.requirement = requirement
        self.technical_spec = technical_spec
        self.repository = repository
        self.selection_service = selection_service
        self.shadow_runner = shadow_runner
        self.policy_consumer = policy_consumer
        self.get_current_revision = get_current_revision
        self.shadow_timeout_seconds = shadow_timeout_seconds
        self.context_builder = ProjectContextBuilder()

    async def build(
        self,
        task: TaskDefinition,
    ) -> TaskContextBundle:

        revision = self.get_current_revision()

        # OFF 模式不执行 Shadow/Rerank。
        if self.policy_consumer.config.mode == "off":
            result = await self.selection_service.select(
                repository_root=self.root,
                repository_revision=revision,
                task=task,
                raw_requirement=self.raw_requirement,
                requirement_spec=self.requirement,
            )
            report = {}
            fresh_report = False

        else:
            reports = []

            wrapper = ShadowTaskContextSelectionService(
                delegate=self.selection_service,
                config=ShadowPolicyConfig(
                    mode="shadow",
                    timeout_seconds=self.shadow_timeout_seconds,
                ),
                runner=self.shadow_runner,
                emit=reports.append,
            )

            result = await wrapper.select(
                repository_root=self.root,
                repository_revision=revision,
                task=task,
                raw_requirement=self.raw_requirement,
                requirement_spec=self.requirement,
            )

            report = reports[0] if reports else {}
            fresh_report = bool(reports)

        def build_context(selection):
            return self.context_builder.build(
                repository_root=self.root,
                requirement=self.requirement,
                technical_spec=self.technical_spec,
                task=task,
                repository=self.repository,
                selection=selection,
                max_files=8,
                max_total_chars=40_000,
                max_chars_per_file=10_000,
            )

        consumption = self.policy_consumer.consume(
            repository_root=self.root,
            task=task,
            raw_requirement=self.raw_requirement,
            selection_result=result,
            report=report,
            get_current_revision=self.get_current_revision,
            build_context=build_context,
            fresh_in_process_report=fresh_report,
        )

        return consumption.bundle
