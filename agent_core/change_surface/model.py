from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ObservedFileRole = Literal[
    "candidate_target",
    "dependency",
    "reference",
    "test",
    "unknown",
]

ExecutionDisposition = Literal[
    "change_required",
    "no_change_required",
    "insufficient_evidence",
]


class ObservedRepositoryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str = Field(min_length=1)
    observed_role: ObservedFileRole
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_evidence(self) -> "ObservedRepositoryFile":
        if self.observed_role != "unknown" and not self.evidence:
            raise ValueError("非 unknown 的 observed_role 必须至少包含一条 evidence")
        return self


class InvestigationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    observed_files: list[ObservedRepositoryFile] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_files(self) -> "InvestigationArtifact":
        paths = [item.relative_path for item in self.observed_files]
        if len(paths) != len(set(paths)):
            raise ValueError("InvestigationArtifact observed_files 中存在重复文件")
        return self


class ResolvedChangeSurface(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)

    implementation_targets: list[str] = Field(default_factory=list)

    dependencies: list[str] = Field(default_factory=list)

    references: list[str] = Field(default_factory=list)

    tests: list[str] = Field(default_factory=list)

    # ==================================
    # Prospective Write Targets
    #
    # 当前文件还不存在，
    # 但已经由 Harness 批准
    # 可以 CREATE。
    #
    # v1 只允许 Test Task 使用。
    #
    # create_files 必须同时属于 tests。
    # ==================================

    create_files: list[str] = Field(default_factory=list)

    unresolved: list[str] = Field(default_factory=list)

    rationale: list[str] = Field(default_factory=list)

    execution_disposition: ExecutionDisposition | None = Field(
        default=None,
        description=(
            "Runtime 根据最终 "
            "Change Surface "
            "推导的执行处置："
            "需要修改、无需修改、"
            "或证据不足。"
        ),
    )

    @model_validator(mode="after")
    def validate_file_roles(
        self,
    ) -> "ResolvedChangeSurface":

        groups = {
            "implementation_targets": (self.implementation_targets),
            "dependencies": (self.dependencies),
            "references": (self.references),
            "tests": (self.tests),
        }

        owner_by_path: dict[
            str,
            str,
        ] = {}

        for role, paths in groups.items():
            for path in paths:
                normalized = path.replace(
                    "\\",
                    "/",
                ).strip()

                if not normalized:
                    raise ValueError(f"{role} " "中不能包含空路径")

                previous = owner_by_path.get(normalized)

                if previous is not None:
                    raise ValueError(
                        f"文件 {normalized} " f"同时出现在 " f"{previous} " f"和 {role}"
                    )

                owner_by_path[normalized] = role

        normalized_create = [
            path.replace("\\", "/").strip() for path in self.create_files
        ]

        if any(not path for path in normalized_create):
            raise ValueError("create_files " "不能包含空路径")

        if len(normalized_create) != len(set(normalized_create)):
            raise ValueError("create_files " "存在重复路径")

        normalized_tests = {path.replace("\\", "/").strip() for path in self.tests}

        missing_test_role = set(normalized_create) - normalized_tests

        if missing_test_role:
            raise ValueError(
                "v1 create_files "
                "必须同时属于 tests: " + ", ".join(sorted(missing_test_role))
            )

        non_test_roles = {
            path.replace("\\", "/").strip()
            for path in [
                *self.implementation_targets,
                *self.dependencies,
                *self.references,
            ]
        }

        overlap = set(normalized_create) & non_test_roles

        if overlap:
            raise ValueError(
                "create file "
                "不能同时属于生产/"
                "dependency/reference: " + ", ".join(sorted(overlap))
            )

        return self

    def allowed_change_files(
        self,
    ) -> set[str]:
        return {
            *self.implementation_targets,
            *self.tests,
        }

    def requires_patch(
        self,
    ) -> bool:
        return self.execution_disposition == "change_required" and bool(
            self.implementation_targets or self.tests
        )


class ScopeGuardResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    allowed: bool
    requested_files: list[str] = Field(default_factory=list)
    allowed_files: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)


class RepositoryAwareTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str

    task_type: str

    title: str

    objective: str = ""

    depends_on: list[str] = Field(default_factory=list)

    # ==================================
    # Logical Plan Provenance
    # ==================================

    source_planning_stage: Literal[
        "legacy",
        "logical",
    ] = "legacy"

    responsibilities: list[str] = Field(default_factory=list)

    investigation_questions: list[str] = Field(default_factory=list)

    acceptance_criteria_refs: list[str] = Field(default_factory=list)

    # ==================================
    # Initial Repository Hints
    #
    # 只是前置搜索提示，
    # 不等于 resolved targets。
    # ==================================

    original_repository_hints: list[str] = Field(default_factory=list)

    # Legacy compatibility
    original_target_modules: list[str] = Field(default_factory=list)

    # ==================================
    # Evidence-resolved Repository Plan
    # ==================================

    resolved_target_files: list[str] = Field(default_factory=list)

    dependency_files: list[str] = Field(default_factory=list)

    reference_files: list[str] = Field(default_factory=list)

    test_files: list[str] = Field(default_factory=list)

    repository_revision: str | None = None

    resolution_status: Literal[
        "resolved",
        "partial",
        "unresolved",
    ]

    execution_disposition: ExecutionDisposition | None = None

    reasons: list[str] = Field(default_factory=list)


class RepositoryAwareTaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str

    source_planning_stage: Literal[
        "legacy",
        "logical",
    ] = "legacy"

    repository_revision: str | None = None

    tasks: list[RepositoryAwareTask] = Field(default_factory=list)
