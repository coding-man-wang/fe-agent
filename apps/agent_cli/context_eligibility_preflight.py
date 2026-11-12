"""Read a saved Full Execution run without invoking models."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.benchmark.execution_model import FullExecutionRunRecord
from agent_core.benchmark.context_eligibility_preflight import (
    analyze_context_eligibility,
)
from agent_core.change_surface.model import ResolvedChangeSurface


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", default="TASK-002")
    parser.add_argument("--fe-agent-root", default=".")
    args = parser.parse_args()

    if re.fullmatch(r"[0-9a-f]{32}", args.run_id) is None:
        parser.error("Invalid run ID")

    root = Path(args.fe_agent_root).resolve()
    base = root / ".fe-agent" / "benchmark" / "full_execution"

    run_path = base / "runs" / f"{args.run_id}.json"
    run = FullExecutionRunRecord.model_validate_json(
        run_path.read_text(encoding="utf-8-sig")
    )

    if run.run_id != args.run_id:
        raise RuntimeError("Run ID mismatch")

    task = next(
        (t for t in run.tasks if t.task_id == args.task_id),
        None,
    )

    def emit_inconclusive(reason: str) -> None:
        print(
            json.dumps(
                {
                    "version": "v1.4.9-P4-B-D3",
                    "status": "inconclusive",
                    "reason": reason,
                    "run_id": run.run_id,
                    "task_id": args.task_id,
                    "historical_advisory_only": True,
                    "applied": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if task is None:
        emit_inconclusive("task_record_missing")
        return

    guard = task.context_guarded_active or {}
    context_files = task.selected_context_files

    if (
        guard.get("status") not in {"fallback", "active"}
        or guard.get("task_id") != task.task_id
        or guard.get("case_id") != run.case_id
        or guard.get("repository_revision") != run.base_commit_sha
        or guard.get("baseline_files") != context_files
    ):
        emit_inconclusive("context_provenance_unverified")
        return

    artifact_root = base / run.run_id / "harness" / "artifacts"
    artifact = JsonTaskArtifactStore(root=artifact_root).latest(
        task_id=args.task_id,
        artifact_type="change_surface",
    )

    if artifact is None:
        emit_inconclusive("surface_artifact_missing")
        return

    # A seeded historical surface is not fresh evidence.
    if artifact.created_at < run.started_at:
        emit_inconclusive("surface_not_created_in_this_run")
        return

    surface = ResolvedChangeSurface.model_validate(artifact.payload)

    if surface.task_id != task.task_id:
        emit_inconclusive("surface_task_mismatch")
        return

    report = analyze_context_eligibility(
        context_files=context_files,
        surface=surface,
        obligation_evidence_files=task.obligation_evidence_files,
        verification_strategy=task.verification_strategy,
    )

    report["run_id"] = run.run_id
    report["case_id"] = run.case_id
    report["base_commit_sha"] = run.base_commit_sha
    report["surface_artifact_id"] = artifact.artifact_id
    report["surface_created_at"] = artifact.created_at.isoformat()

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
