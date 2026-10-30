import json
from pathlib import Path

import pytest

from agent_core.codebase.exact_search import (
    ExactCodeSearcher,
)
from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.hybrid_relevance import (
    HybridRelevantFileSelector,
)
from agent_core.codebase.hybrid_retrieval import (
    HybridRetriever,
)
from agent_core.codebase.model import (
    CodeChunk,
    RepositoryProfile,
    SemanticSearchResponse,
    SemanticSearchResult,
)
from agent_core.codebase.task_retrieval import (
    TaskRetrievalService,
)
from agent_core.coding.agent import (
    ReadOnlyCodingAgent,
)
from agent_core.coding.checkpoint import (
    AnalysisCheckpointStore,
)
from agent_core.coding.durable_workflow import (
    DurableReadOnlyWorkflow,
)
from agent_core.coding.handoff import (
    AnalysisArtifactStore,
)
from agent_core.context.budget_provider import (
    StaticContextBudgetProvider,
)
from agent_core.context.builder import (
    ProjectContextBuilder,
)
from agent_core.context.factory import (
    DefaultTaskContextFactory,
)
from agent_core.context.model import (
    ContextBudgetConfig,
)
from agent_core.context.rules_provider import (
    StaticEngineeringRulesProvider,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


class FakeSemanticSearcher:

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> SemanticSearchResponse:
        result = SemanticSearchResult(
            chunk=CodeChunk(
                chunk_id=("src/api/user.ts:1-3"),
                relative_path=("src/api/user.ts"),
                start_line=1,
                end_line=3,
                content=("export interface UserQuery {\n" "  page: number\n" "}\n"),
                language="TypeScript",
                category="source",
            ),
            score=0.95,
        )

        return SemanticSearchResponse(
            query=query,
            results=[result][:top_k],
        )


class ChangeSurfaceAwareContextFactory:
    """
    测试适配器。

    DefaultTaskContextFactory 负责构建
    Retrieval / Code Context，

    当前测试再模拟已经由上游
    Investigation + Change Surface
    确认：

        src/api/user.ts

    是当前 Task 的合法写目标。

    注意：
    这不是从 Retrieval 推导写权限，
    而是显式模拟上游 Artifact。
    """

    def __init__(
        self,
        delegate,
    ):
        self.delegate = delegate

    async def build(
        self,
        task,
    ):
        bundle = await self.delegate.build(task)

        return bundle.model_copy(
            deep=True,
            update={
                "change_surface": {
                    "task_id": task.task_id,
                    "implementation_targets": [
                        "src/api/user.ts",
                    ],
                    "dependencies": [],
                    "references": [],
                    "tests": [],
                    "create_files": [],
                    "unresolved": [],
                    "rationale": [
                        (
                            "测试夹具模拟上游 "
                            "Change Surface 已确认 "
                            "src/api/user.ts "
                            "为当前 writable target"
                        )
                    ],
                    "execution_disposition": "change_required",
                }
            },
        )


class FakeCodingModel:
    """
    Fake 的不是 CodingAgent，
    而只是最底层 LLM。

    这样 ReadOnlyCodingAgent 自己的：
    - Message Preparation
    - JSON Validation
    - Evidence Validation

    都会真实运行。
    """

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str:
        return json.dumps(
            {
                "task_id": "TASK-001",
                "summary": ("用户列表查询需要增加 " "status 筛选能力"),
                "findings": [
                    {
                        "relative_path": ("src/api/user.ts"),
                        "evidence": ("export interface " "UserQuery"),
                        "explanation": ("当前查询参数类型中" "尚未定义 status"),
                    }
                ],
                "proposed_changes": [
                    {
                        "relative_path": ("src/api/user.ts"),
                        "objective": ("增加 status 查询参数"),
                        "proposed_approach": (
                            "扩展 UserQuery 类型，" "增加可选 status 字段"
                        ),
                    }
                ],
                "verification_plan": [
                    "运行 TypeScript 类型检查",
                    "运行用户列表相关测试",
                ],
                "missing_information": [],
                "execution_status": ("not_executed"),
            },
            ensure_ascii=False,
        )


def create_repository(
    root: Path,
) -> None:
    api = root / "src" / "api"

    api.mkdir(parents=True)

    (api / "user.ts").write_text(
        (
            "export interface UserQuery {\n"
            "  page: number\n"
            "}\n\n"
            "export async function "
            "getUserList("
            "query: UserQuery"
            ") {}\n"
        ),
        encoding="utf-8",
    )


def create_requirement() -> RequirementSpec:
    return RequirementSpec(
        title="用户状态筛选",
        summary=("用户列表支持 " "status 筛选"),
        goals=[
            "支持 status 查询",
        ],
        acceptance_criteria=[
            ("用户列表能够按 " "status 过滤"),
        ],
    )


def create_technical_spec() -> TechnicalSpec:
    return TechnicalSpec(
        summary=("扩展用户列表查询参数"),
        solution_approach=[
            ("为 UserQuery 增加 " "status 参数"),
        ],
        affected_modules=[
            "src/api",
        ],
        testing_strategy=[
            "运行类型检查",
        ],
    )


def create_task() -> TaskDefinition:
    return TaskDefinition(
        task_id="TASK-001",
        task_type="code_change",
        title="扩展用户查询参数",
        objective=("给用户列表 API " "增加 status 筛选"),
        source_step_order=1,
        depends_on=[],
        inputs=[
            "现有 UserQuery",
        ],
        target_modules=[
            "src/api",
        ],
        expected_outputs=[
            ("UserQuery 支持 " "status"),
        ],
        verification=[
            "类型检查通过",
        ],
        acceptance_criteria_refs=[],
    )


@pytest.mark.asyncio
async def test_analysis_workflow_end_to_end(
    tmp_path: Path,
):
    # ------------------------------------------------
    # 1. 准备真实测试 Repository
    # ------------------------------------------------

    create_repository(tmp_path)

    index = RepositoryFileIndexer().build(tmp_path)

    repository = RepositoryProfile(
        project_name="user-admin",
        root_path=str(tmp_path),
        framework="Vue3",
        language="TypeScript",
        build_tool="Vite",
        package_manager="pnpm",
    )

    # ------------------------------------------------
    # 2. 真实 Retrieval Pipeline
    # ------------------------------------------------

    retrieval_service = TaskRetrievalService(
        exact_searcher=(ExactCodeSearcher()),
        semantic_searcher=(FakeSemanticSearcher()),
        hybrid_retriever=(HybridRetriever()),
        selector=(HybridRelevantFileSelector()),
    )

    # ------------------------------------------------
    # 3. 真实 Context Factory
    # ------------------------------------------------

    base_context_factory = DefaultTaskContextFactory(
        repository_root=tmp_path,
        requirement=(create_requirement()),
        technical_spec=(create_technical_spec()),
        repository=repository,
        index=index,
        retrieval_service=(retrieval_service),
        context_builder=(ProjectContextBuilder()),
        rules_provider=(
            StaticEngineeringRulesProvider(
                [
                    ("保持现有 API " "向后兼容"),
                ]
            )
        ),
        budget_provider=(
            StaticContextBudgetProvider(
                ContextBudgetConfig(
                    max_files=5,
                    max_total_chars=20_000,
                    max_chars_per_file=8_000,
                )
            )
        ),
    )

    context_factory = ChangeSurfaceAwareContextFactory(base_context_factory)

    # ------------------------------------------------
    # 4. 真实 ReadOnlyCodingAgent
    # ------------------------------------------------

    analyzer = ReadOnlyCodingAgent(
        model=FakeCodingModel(),
    )

    # ------------------------------------------------
    # 5. 真实 Durable Stores
    # ------------------------------------------------

    checkpoint_store = AnalysisCheckpointStore(tmp_path)

    artifact_store = AnalysisArtifactStore(tmp_path)

    # ------------------------------------------------
    # 6. 真实 Durable Workflow
    # ------------------------------------------------

    workflow = DurableReadOnlyWorkflow(
        context_factory=(context_factory),
        analyzer=analyzer,
        checkpoint_store=(checkpoint_store),
        artifact_store=(artifact_store),
    )

    task = create_task()

    # ------------------------------------------------
    # 7. 真正执行整条 Analysis Workflow
    # ------------------------------------------------

    record = await workflow.run_next(
        tasks=[task],
        completed_task_ids=set(),
    )

    # ------------------------------------------------
    # 8. 验证 Workflow Result
    # ------------------------------------------------

    assert record is not None

    assert record.status == "succeeded", record.error

    assert record.result is not None

    assert record.result.task_id == "TASK-001"

    assert record.result.execution_status == "not_executed"

    assert record.artifact_id is not None

    assert len(record.artifact_id) == 64

    assert record.context_hash is not None

    assert len(record.context_hash) == 64

    # ------------------------------------------------
    # 9. 验证 Artifact 真正落盘
    # ------------------------------------------------

    artifact = artifact_store.load(record.artifact_id)

    assert artifact.task_id == "TASK-001"

    assert artifact.context_hash == record.context_hash

    assert artifact.result == record.result

    # ------------------------------------------------
    # 10. 验证 Checkpoint 真正落盘
    # ------------------------------------------------

    checkpoint = checkpoint_store.load()

    assert "TASK-001" in checkpoint.records

    persisted = checkpoint.records["TASK-001"]

    assert persisted.status == "succeeded"

    assert persisted.artifact_id == record.artifact_id

    assert persisted.context_hash == record.context_hash


class InvalidEvidenceModel:

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str:
        return json.dumps(
            {
                "task_id": "TASK-001",
                "summary": ("错误分析结果"),
                "findings": [
                    {
                        "relative_path": ("src/api/user.ts"),
                        "evidence": ("THIS CODE " "DOES NOT EXIST"),
                        "explanation": ("伪造证据"),
                    }
                ],
                "proposed_changes": [],
                "verification_plan": [],
                "missing_information": [],
                "execution_status": ("not_executed"),
            }
        )
