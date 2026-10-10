from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

from agent_core.coding.handoff import (
    write_json_atomically,
)
from agent_core.coding.orchestration_model import (
    TaskAnalysisRecord,
)


class AnalysisCheckpoint(BaseModel):
    schema_version: Literal[1] = 1

    records: dict[
        str,
        TaskAnalysisRecord,
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_records(
        self,
    ) -> "AnalysisCheckpoint":

        for task_id, record in self.records.items():

            if record.task_id != task_id:
                raise ValueError("Checkpoint 任务键与记录 ID 不一致")

            if record.status == "succeeded":
                if (record.artifact_id is None) != (record.context_hash is None):
                    raise ValueError("Artifact ID 和 Context Hash 必须成对出现")

                if record.result is None:
                    raise ValueError("成功的分析记录缺少结果")

                if record.result.task_id != task_id:
                    raise ValueError("分析结果的 Task ID 不一致")

        return self


class AnalysisCheckpointStore:

    def __init__(
        self,
        root: str | Path,
    ) -> None:
        self.root = Path(root).resolve()

        self.path = self.root / ".fe-agent" / "runtime" / "checkpoint.json"

    def save(
        self,
        records: dict[
            str,
            TaskAnalysisRecord,
        ],
    ) -> None:
        checkpoint = AnalysisCheckpoint(records=records)

        write_json_atomically(
            self.path,
            checkpoint.model_dump_json(indent=2),
        )

    def load(
        self,
    ) -> AnalysisCheckpoint:
        if not self.path.exists():
            return AnalysisCheckpoint()

        return AnalysisCheckpoint.model_validate_json(
            self.path.read_text(encoding="utf-8")
        )
