import json

import pytest

from agent_core.codebase.model import (
    RepositoryProfile,
)
from agent_core.coding.agent import (
    CodingOutputError,
    ReadOnlyCodingAgent,
)
from agent_core.context.model import (
    CodeContextItem,
    ContextBudgetUsage,
    TaskContextBundle,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)

SOURCE = "export interface UserQuery {\n" "  page: number\n" "}\n"


def create_bundle() -> TaskContextBundle:

    return TaskContextBundle(
        requirement=RequirementSpec(
            title="用户列表",
            summary="支持用户列表查询",
        ),
        technical_spec=TechnicalSpec(
            summary="扩展用户查询能力",
        ),
        task=TaskDefinition(
            task_id="TASK-001",
            task_type="code_change",
            title="修改用户查询",
            objective="扩展用户查询能力",
            source_step_order=1,
            responsibilities=[
                "维护用户查询能力",
            ],
            repository_hints=[
                "src/api",
            ],
            target_modules=[],
        ),
        repository=RepositoryProfile(
            project_name="demo",
            root_path=".",
            framework="Vue3",
            language="TypeScript",
        ),
        code_context=[
            CodeContextItem(
                relative_path=("src/api/user.ts"),
                category="source",
                language="TypeScript",
                content_mode="full_file",
                content=SOURCE,
                start_line=1,
                end_line=3,
                content_chars=len(SOURCE),
            )
        ],
        budget=ContextBudgetUsage(
            max_files=5,
            max_total_chars=10_000,
            max_chars_per_file=5_000,
            used_files=1,
            used_chars=len(SOURCE),
        ),
        change_surface={
            "implementation_targets": [
                "src/api/user.ts",
            ],
            "dependencies": [],
            "references": [],
            "tests": [],
            "create_files": [],
        },
    )


def coding_result(
    *,
    evidence: str,
    proposed_changes=None,
) -> str:

    if proposed_changes is None:
        proposed_changes = [
            {
                "relative_path": "src/api/user.ts",
                "action": "modify",
                "objective": "扩展用户查询接口",
                "proposed_approach": "扩展现有 UserQuery",
            }
        ]

    return json.dumps(
        {
            "task_id": "TASK-001",
            "summary": "UserQuery 需要扩展",
            "findings": [
                {
                    "relative_path": "src/api/user.ts",
                    "evidence": evidence,
                    "explanation": "当前查询参数由 " "UserQuery 定义",
                }
            ],
            "proposed_changes": proposed_changes,
            "verification_plan": [
                "运行 typecheck",
            ],
            "missing_information": [],
            "evidence_gaps": [],
            "execution_status": "not_executed",
        },
        ensure_ascii=False,
    )


class CorrectableEvidenceModel:

    def __init__(self):
        self.call_count = 0

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str:

        self.call_count += 1

        if self.call_count == 1:
            return coding_result(evidence=("export interface " "UserQueryFake"))

        return coding_result(evidence=("export interface " "UserQuery"))


@pytest.mark.asyncio
async def test_invalid_evidence_is_corrected_once():

    model = CorrectableEvidenceModel()

    agent = ReadOnlyCodingAgent(model=model)

    result = await agent.analyze(create_bundle())

    assert model.call_count == 2

    assert result.findings[0].evidence == "export interface UserQuery"

    assert result.proposed_changes[0].relative_path == "src/api/user.ts"


class PersistentInvalidEvidenceModel:

    def __init__(self):
        self.call_count = 0

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str:

        self.call_count += 1

        return coding_result(evidence=("THIS CODE DOES NOT EXIST"))


@pytest.mark.asyncio
async def test_evidence_correction_is_bounded():

    model = PersistentInvalidEvidenceModel()

    agent = ReadOnlyCodingAgent(model=model)

    with pytest.raises(
        CodingOutputError,
        match=("Bounded Evidence Correction"),
    ):
        await agent.analyze(create_bundle())

    # 初次分析 + 1 次纠正。
    #
    # 不允许无限 Retry。
    assert model.call_count == 2


class DecisionMutationModel:

    def __init__(self):
        self.call_count = 0

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str:

        self.call_count += 1

        if self.call_count == 1:
            return coding_result(evidence=("THIS CODE DOES NOT EXIST"))

        # Evidence 修好了，
        # 但是偷偷删除了 change。
        return coding_result(
            evidence=("export interface " "UserQuery"),
            proposed_changes=[],
        )


@pytest.mark.asyncio
async def test_evidence_correction_cannot_change_decision():

    model = DecisionMutationModel()

    agent = ReadOnlyCodingAgent(model=model)

    with pytest.raises(
        CodingOutputError,
        match="proposed_changes",
    ):
        await agent.analyze(create_bundle())

    assert model.call_count == 2
