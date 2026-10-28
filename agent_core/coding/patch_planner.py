from agent_core.coding.handoff import (
    AnalysisArtifact,
)

from agent_core.coding.patch_model import (
    PatchPlan,
    PatchTarget,
)

from agent_core.coding.snapshot import (
    TaskFileSnapshot,
)


class PatchPlanningError(ValueError):
    pass


class PatchPlanner:

    def plan(
        self,
        *,
        artifact: AnalysisArtifact,
        snapshot: TaskFileSnapshot,
    ) -> PatchPlan:

        if artifact.task_id != snapshot.task_id:
            raise PatchPlanningError("Artifact 与 Snapshot " "不属于同一个 Task")

        snapshot_map = {item.relative_path: item for item in snapshot.files}

        targets: list[PatchTarget] = []

        seen_paths: set[str] = set()

        for change in artifact.result.proposed_changes:

            path = change.relative_path

            if path in seen_paths:
                raise PatchPlanningError("同一文件存在" "重复修改提案: " f"{path}")

            seen_paths.add(path)

            if change.action == "modify":

                file_snapshot = snapshot_map.get(path)

                if file_snapshot is None:
                    raise PatchPlanningError(
                        "modify target " "不存在于分析快照: " f"{path}"
                    )

                expected_sha256 = file_snapshot.sha256

            elif change.action == "create":

                if path in snapshot_map:
                    raise PatchPlanningError(
                        "create target " "已经存在于分析快照: " f"{path}"
                    )

                expected_sha256 = None

            else:
                raise PatchPlanningError("未知 Patch Action: " f"{change.action}")

            targets.append(
                PatchTarget(
                    relative_path=path,
                    action=(change.action),
                    expected_sha256=(expected_sha256),
                    objective=(change.objective),
                    proposed_approach=(change.proposed_approach),
                )
            )

        targets.sort(key=lambda item: item.relative_path)

        return PatchPlan(
            task_id=artifact.task_id,
            artifact_id=(artifact.artifact_id),
            context_hash=(artifact.context_hash),
            targets=targets,
        )
