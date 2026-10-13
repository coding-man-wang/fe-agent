import re
from pathlib import Path

from agent_core.coding.handoff import (
    write_json_atomically,
)
from agent_core.git.revision_workflow_state import (
    DurableRevisionWorkflowState,
)


class DurableRevisionStoreError(ValueError):
    pass


class DurableRevisionWorkflowStore:

    def __init__(
        self,
        repository_root: str | Path,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        self.directory = self.root / ".fe-agent" / "runtime" / "revision_workflows"

    def _path(
        self,
        workflow_id: str,
    ) -> Path:

        if not re.fullmatch(
            r"[0-9a-f]{64}",
            workflow_id,
        ):
            raise (DurableRevisionStoreError("workflow_id 非法"))

        return self.directory / f"{workflow_id}.json"

    def save(
        self,
        state: DurableRevisionWorkflowState,
    ) -> None:

        path = self._path(state.workflow_id)

        write_json_atomically(
            path,
            state.model_dump_json(indent=2),
        )

    def load(
        self,
        workflow_id: str,
    ) -> DurableRevisionWorkflowState:

        path = self._path(workflow_id)

        if not path.is_file():

            raise (DurableRevisionStoreError("Revision Workflow " "不存在"))

        try:

            state = DurableRevisionWorkflowState.model_validate_json(
                path.read_text(encoding="utf-8")
            )

        except Exception as exc:

            raise (DurableRevisionStoreError("Revision Workflow " "记录非法")) from exc

        if state.workflow_id != workflow_id:

            raise (DurableRevisionStoreError("Workflow ID 不一致"))

        return state
