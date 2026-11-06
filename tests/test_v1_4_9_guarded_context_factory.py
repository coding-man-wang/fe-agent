from types import SimpleNamespace

import pytest

from agent_core.benchmark.context_policy_runtime import (
    ContextRolloutConfig,
    GuardedContextPolicyConsumer,
)
from agent_core.benchmark.guarded_context_factory import (
    GuardedTaskContextFactory,
)
from agent_core.codebase.model import (
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryProfile,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


class FakeSelectionService:
    def __init__(self, selection):
        self.calls = 0
        self.selection = selection

    async def select(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            selection=self.selection,
            retrieval_selection=self.selection,
            repository_revision=kwargs["repository_revision"],
        )


class FakeShadowRunner:
    def __init__(self):
        self.calls = 0

    async def observe(self, *, result, **kwargs):
        self.calls += 1

        return {
            "status": "complete",
            "baseline_file_count": 1,
            "proposed_file_count": 1,
            "graph_proposed_drops": [],
            "rerank_proposed_drops": [],
            "effective_proposed_drops": [],
        }


@pytest.fixture
def factory_case(tmp_path):
    path = tmp_path / "src" / "demo.ts"
    path.parent.mkdir(parents=True)
    path.write_text(
        "export const demo = 1\n",
        encoding="utf-8",
    )

    item = HybridRelevantFile(
        relative_path="src/demo.ts",
        category="source",
        language="TypeScript",
        size_bytes=path.stat().st_size,
        retrieval_score=1.0,
        retrieval_rank=1,
        policy_score=0,
        selection_score=1.0,
    )

    selection = HybridRelevantFileSelection(
        candidate_count=1,
        eligible_count=1,
        selected_count=1,
        total_selected_bytes=item.size_bytes,
        selected_files=[item],
    )

    task = TaskDefinition(
        task_id="TASK-001",
        task_type="code_change",
        title="修改 demo",
        objective="修改 demo",
        source_step_order=1,
    )

    return {
        "root": tmp_path,
        "task": task,
        "delegate": FakeSelectionService(selection),
        "runner": FakeShadowRunner(),
    }


def create_factory(case, *, mode):
    return GuardedTaskContextFactory(
        repository_root=case["root"],
        raw_requirement="修改 demo",
        requirement=RequirementSpec(
            title="修改 demo",
            summary="修改 demo",
        ),
        technical_spec=TechnicalSpec(
            summary="调整 demo 实现",
        ),
        repository=RepositoryProfile(
            project_name="demo",
            root_path=str(case["root"]),
            language="TypeScript",
        ),
        selection_service=case["delegate"],
        shadow_runner=case["runner"],
        policy_consumer=GuardedContextPolicyConsumer(
            config=ContextRolloutConfig(mode=mode),
        ),
        get_current_revision=lambda: "revision-001",
    )


@pytest.mark.asyncio
async def test_factory_off_skips_shadow(factory_case):
    factory = create_factory(
        factory_case,
        mode="off",
    )

    bundle = await factory.build(factory_case["task"])

    assert bundle.task.task_id == "TASK-001"
    assert len(bundle.code_context) == 1
    assert factory_case["delegate"].calls == 1
    assert factory_case["runner"].calls == 0


@pytest.mark.asyncio
async def test_factory_shadow_returns_baseline(
    factory_case,
):
    factory = create_factory(
        factory_case,
        mode="shadow",
    )

    bundle = await factory.build(factory_case["task"])

    assert bundle.task.task_id == "TASK-001"
    assert [item.relative_path for item in bundle.code_context] == ["src/demo.ts"]

    assert factory_case["delegate"].calls == 1
    assert factory_case["runner"].calls == 1
