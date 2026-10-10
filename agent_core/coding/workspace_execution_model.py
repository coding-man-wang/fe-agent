from pydantic import (
    BaseModel,
    ConfigDict,
)

from agent_core.coding.execution_journal_model import (
    PatchExecutionJournal,
)
from agent_core.coding.task_completion_model import (
    TaskCompletionRecord,
)
from agent_core.coding.verification_record_model import (
    VerificationRunRecord,
)


class WorkspaceExecutionResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    workspace_id: str

    task_id: str

    proposal_id: str

    execution_root: str

    journal: PatchExecutionJournal

    verification: VerificationRunRecord

    completion: TaskCompletionRecord
