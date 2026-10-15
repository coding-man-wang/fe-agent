from datetime import (
    datetime,
    timezone,
)
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)

from agent_core.benchmark.evaluation_model import (
    BenchmarkRetrievalEvaluation,
)
from agent_core.benchmark.retrieval_model import (
    BenchmarkRetrievalCapture,
)
from agent_core.sdd.model import (
    ImplementationPlan,
    ProjectContext,
    RequirementSpec,
    TaskPlan,
    TechnicalSpec,
)

BenchmarkRunStage = Literal[
    "baseline_verified",
    "requirement_analyzed",
    "technical_designed",
    "implementation_planned",
    "task_planned",
    "retrieval_captured",
    "retrieval_evaluated",
]


class BenchmarkRunRecord(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[1] = 1

    run_id: str

    case_id: str

    repository_root: str

    base_commit_sha: str

    status: Literal[
        "running",
        "completed",
        "failed",
    ]

    stage: BenchmarkRunStage

    started_at: datetime

    finished_at: datetime | None = None

    project_context: ProjectContext | None = None

    requirement_spec: RequirementSpec | None = None

    technical_spec: TechnicalSpec | None = None

    implementation_plan: ImplementationPlan | None = None

    task_plan: TaskPlan | None = None

    retrieval_capture: BenchmarkRetrievalCapture | None = None

    retrieval_evaluation: BenchmarkRetrievalEvaluation | None = None

    error: str | None = None


def utc_now() -> datetime:

    return datetime.now(timezone.utc)
