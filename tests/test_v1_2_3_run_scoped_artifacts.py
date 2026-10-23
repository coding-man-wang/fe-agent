from agent_core.artifacts.model import (
    TaskArtifact,
)
from agent_core.artifacts.store import (
    JsonTaskArtifactStore,
)


def test_run_scoped_store_does_not_seed_execution(
    tmp_path,
):
    baseline = JsonTaskArtifactStore(root=tmp_path / "baseline")

    run_store = JsonTaskArtifactStore(root=tmp_path / "run")

    baseline.put(
        TaskArtifact(
            task_id="TASK-001",
            artifact_type="investigation",
            payload={"marker": "baseline-investigation"},
        )
    )

    # 模拟历史 Run 曾经留下的 Execution。
    #
    # 新 Run 绝对不能继承它。
    baseline.put(
        TaskArtifact(
            task_id="TASK-002",
            artifact_type="execution",
            payload={"marker": "stale-execution"},
        )
    )

    copied = run_store.seed_from(
        source=baseline,
        artifact_types={
            "investigation",
            "change_surface",
            "replan",
        },
    )

    assert copied == 1

    investigation = run_store.latest(
        task_id="TASK-001",
        artifact_type="investigation",
    )

    assert investigation is not None

    assert investigation.payload["marker"] == "baseline-investigation"

    stale_execution = run_store.latest(
        task_id="TASK-002",
        artifact_type="execution",
    )

    assert stale_execution is None


def test_run_execution_does_not_pollute_baseline(
    tmp_path,
):
    baseline = JsonTaskArtifactStore(root=tmp_path / "baseline")

    run_store = JsonTaskArtifactStore(root=tmp_path / "run")

    baseline.put(
        TaskArtifact(
            task_id="TASK-001",
            artifact_type="investigation",
            payload={"marker": "baseline"},
        )
    )

    run_store.seed_from(
        source=baseline,
        artifact_types={
            "investigation",
            "change_surface",
            "replan",
        },
    )

    run_store.put(
        TaskArtifact(
            task_id="TASK-002",
            artifact_type="execution",
            payload={"marker": "current-run"},
        )
    )

    assert (
        run_store.latest(
            task_id="TASK-002",
            artifact_type="execution",
        )
        is not None
    )

    assert (
        baseline.latest(
            task_id="TASK-002",
            artifact_type="execution",
        )
        is None
    )
