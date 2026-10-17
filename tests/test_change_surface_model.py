import pytest
from pydantic import ValidationError

from agent_core.change_surface import (
    InvestigationArtifact,
    ObservedRepositoryFile,
    ResolvedChangeSurface,
)


def test_observed_repository_file_accepts_candidate_target():
    item = ObservedRepositoryFile(
        relative_path=("src/stores/users.ts"),
        observed_role=("candidate_target"),
        evidence=[
            ("文件定义用户列表查询状态，" "其中包含 pageSize"),
            ("pageSize 会参与用户列表" "分页数据加载"),
        ],
        confidence=0.95,
    )

    assert item.relative_path == "src/stores/users.ts"

    assert item.observed_role == "candidate_target"

    assert item.confidence == 0.95


def test_observed_repository_file_requires_evidence_for_known_role():
    with pytest.raises(ValidationError):
        ObservedRepositoryFile(
            relative_path=("src/stores/users.ts"),
            observed_role=("candidate_target"),
            evidence=[],
            confidence=0.9,
        )


def test_unknown_observed_role_can_have_no_evidence():
    item = ObservedRepositoryFile(
        relative_path=("src/components/" "UnknownWidget.vue"),
        observed_role="unknown",
        evidence=[],
        confidence=0.2,
    )

    assert item.observed_role == "unknown"


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        ObservedRepositoryFile(
            relative_path=("src/stores/users.ts"),
            observed_role="dependency",
            evidence=["被 UsersView 间接使用"],
            confidence=1.5,
        )


def test_investigation_artifact_can_store_repository_findings():
    artifact = InvestigationArtifact(
        task_id="TASK-001",
        summary=("pageSize 当前由 users store " "维护，并被 UsersView 使用。"),
        observed_files=[
            ObservedRepositoryFile(
                relative_path=("src/views/" "UsersView.vue"),
                observed_role=("candidate_target"),
                evidence=["UsersView 组合用户列表" "和分页区域"],
                confidence=0.95,
            ),
            ObservedRepositoryFile(
                relative_path=("src/stores/users.ts"),
                observed_role=("candidate_target"),
                evidence=["users store 已包含 " "pageSize 状态"],
                confidence=0.98,
            ),
            ObservedRepositoryFile(
                relative_path=("src/api/users.ts"),
                observed_role=("dependency"),
                evidence=["现有 API 已接受 " "UserQuery，无需修改协议"],
                confidence=0.9,
            ),
        ],
        findings=[
            ("pageSize 已存在于 " "users store"),
            ("API 请求协议无需修改"),
        ],
        unresolved_questions=["选择控件最终应放在 " "UsersView 还是 " "PaginationBar"],
    )

    assert artifact.task_id == "TASK-001"

    assert len(artifact.observed_files) == 3


def test_investigation_artifact_rejects_duplicate_files():
    with pytest.raises(ValidationError):
        InvestigationArtifact(
            task_id="TASK-001",
            summary="测试重复文件",
            observed_files=[
                ObservedRepositoryFile(
                    relative_path=("src/stores/users.ts"),
                    observed_role=("candidate_target"),
                    evidence=["定义 pageSize"],
                    confidence=0.9,
                ),
                ObservedRepositoryFile(
                    relative_path=("src/stores/users.ts"),
                    observed_role=("dependency"),
                    evidence=["参与查询状态"],
                    confidence=0.8,
                ),
            ],
        )


def test_resolved_change_surface_accepts_unique_roles():
    surface = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=[
            "src/views/UsersView.vue",
            "src/stores/users.ts",
        ],
        dependencies=[
            "src/api/users.ts",
            "src/types/domain.ts",
        ],
        references=["src/components/" "PaginationBar.vue"],
        tests=["tests/unit/" "usersStore.test.ts"],
        unresolved=[],
    )

    assert surface.implementation_targets == [
        "src/views/UsersView.vue",
        "src/stores/users.ts",
    ]

    assert "src/api/users.ts" in surface.dependencies


def test_resolved_change_surface_rejects_file_in_multiple_roles():
    with pytest.raises(ValidationError):
        ResolvedChangeSurface(
            task_id="TASK-002",
            implementation_targets=["src/stores/users.ts"],
            dependencies=["src/stores/users.ts"],
        )


def test_resolved_change_surface_rejects_empty_path():
    with pytest.raises(ValidationError):
        ResolvedChangeSurface(
            task_id="TASK-002",
            implementation_targets=[""],
        )


def test_change_surface_case006_example():
    """
    CASE-006 当前期望形成的
    Change Surface 示例。

    注意：

    这个测试只是验证 Model，
    不是 Benchmark Oracle。

    后续 Resolver 必须通过
    Repository Evidence 自己推导，
    不能直接读取 Benchmark 标准答案。
    """

    surface = ResolvedChangeSurface(
        task_id="TASK-002",
        implementation_targets=[
            "src/views/UsersView.vue",
            "src/stores/users.ts",
        ],
        dependencies=[
            "src/api/users.ts",
            "src/types/domain.ts",
        ],
        references=["src/components/" "PaginationBar.vue"],
        tests=["tests/unit/" "usersStore.test.ts"],
        unresolved=["是否需要新增 E2E 测试" "应由后续测试 Task 决定"],
    )

    assert len(surface.implementation_targets) == 2

    assert len(surface.dependencies) == 2

    assert len(surface.references) == 1

    assert len(surface.tests) == 1
