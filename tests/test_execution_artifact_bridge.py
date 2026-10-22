from agent_core.artifacts.execution_model import (
    TaskExecutionArtifact,
)
from agent_core.artifacts.resolver import (
    TaskArtifactResolver,
)
from agent_core.artifacts.store import (
    JsonTaskArtifactStore,
)
from agent_core.coding.execution_artifact import (
    ExecutionArtifactBuildError,
    TaskExecutionArtifactBuilder,
)
from agent_core.coding.execution_journal_model import (
    FileExecutionRecord,
    PatchExecutionJournal,
)
from agent_core.coding.task_completion_model import (
    TaskCompletionRecord,
)
from agent_core.coding.verification_model import (
    PatchVerificationResult,
    VerificationCheck,
)
from agent_core.coding.verification_record_model import (
    VerificationRunRecord,
)
from agent_core.sdd.model import (
    TaskDefinition,
    TaskPlan,
)

PROPOSAL_ID = "a" * 64
BEFORE_HASH = "b" * 64
AFTER_HASH = "c" * 64


def _execution_artifact() -> TaskExecutionArtifact:
    journal = PatchExecutionJournal(
        proposal_id=PROPOSAL_ID,
        task_id="TASK-002",
        status="applied",
        files=[
            FileExecutionRecord(
                relative_path=("src/stores/users.ts"),
                expected_sha256=(BEFORE_HASH),
                after_sha256=AFTER_HASH,
                status="applied",
            )
        ],
    )

    completion = TaskCompletionRecord(
        task_id="TASK-002",
        proposal_id=PROPOSAL_ID,
    )

    verification = VerificationRunRecord(
        proposal_id=PROPOSAL_ID,
        task_id="TASK-002",
        status="succeeded",
        attempts=1,
        result=(
            PatchVerificationResult(
                proposal_id=(PROPOSAL_ID),
                task_id="TASK-002",
                status="passed",
                checks=[
                    VerificationCheck(
                        name=("frontend-unit-test"),
                        status="passed",
                        details=("all tests passed"),
                    )
                ],
            )
        ),
    )

    return TaskExecutionArtifactBuilder.build(
        journal=journal,
        completion=completion,
        verification=verification,
    )


def _task(
    *,
    task_id: str,
    task_type: str,
    depends_on: list[str],
) -> TaskDefinition:
    return TaskDefinition(
        task_id=task_id,
        task_type=task_type,
        title=task_id,
        objective=f"execute {task_id}",
        source_step_order=1,
        depends_on=depends_on,
        inputs=[],
        target_modules=[],
        expected_outputs=[],
        verification=[],
        acceptance_criteria_refs=[],
    )


def test_build_execution_artifact():
    artifact = _execution_artifact()

    assert artifact.artifact_type == "execution"

    assert artifact.task_id == "TASK-002"

    assert artifact.status == "completed"

    assert artifact.execution_status == "applied"

    assert artifact.verification_status == "passed"

    assert len(artifact.changed_files) == 1

    changed = artifact.changed_files[0]

    assert changed.relative_path == "src/stores/users.ts"

    assert changed.before_sha256 == BEFORE_HASH

    assert changed.after_sha256 == AFTER_HASH


def test_execution_artifact_requires_applied_journal():
    journal = PatchExecutionJournal(
        proposal_id=PROPOSAL_ID,
        task_id="TASK-002",
        status="failed",
        files=[
            FileExecutionRecord(
                relative_path=("src/stores/users.ts"),
                expected_sha256=(BEFORE_HASH),
                after_sha256=AFTER_HASH,
                status="pending",
            )
        ],
        error="apply failed",
    )

    completion = TaskCompletionRecord(
        task_id="TASK-002",
        proposal_id=PROPOSAL_ID,
    )

    verification = VerificationRunRecord(
        proposal_id=PROPOSAL_ID,
        task_id="TASK-002",
        status="succeeded",
        result=(
            PatchVerificationResult(
                proposal_id=(PROPOSAL_ID),
                task_id="TASK-002",
                status="passed",
                checks=[],
            )
        ),
    )

    try:
        (
            TaskExecutionArtifactBuilder.build(
                journal=journal,
                completion=completion,
                verification=verification,
            )
        )
    except ExecutionArtifactBuildError:
        pass
    else:
        raise AssertionError("failed journal " "不应生成 Execution Artifact")


def test_resolver_reads_transitive_upstream_execution(
    tmp_path,
):
    store = JsonTaskArtifactStore(root=tmp_path / "artifacts")

    artifact = _execution_artifact()

    store.put_model(
        task_id="TASK-002",
        artifact_type="execution",
        model=artifact,
        depends_on_task_ids=["TASK-001"],
    )

    task_001 = _task(
        task_id="TASK-001",
        task_type="investigation",
        depends_on=[],
    )

    task_002 = _task(
        task_id="TASK-002",
        task_type="code_change",
        depends_on=["TASK-001"],
    )

    task_003 = _task(
        task_id="TASK-003",
        task_type="code_change",
        depends_on=["TASK-002"],
    )

    task_plan = TaskPlan(
        summary="test plan",
        tasks=[
            task_001,
            task_002,
            task_003,
        ],
        completion_criteria=[],
    )

    resolver = TaskArtifactResolver(store=store)

    executions = resolver.upstream_executions(
        task=task_003,
        task_plan=task_plan,
    )

    assert len(executions) == 1

    assert executions[0].task_id == "TASK-002"

    assert executions[0].changed_files[0].relative_path == "src/stores/users.ts"
