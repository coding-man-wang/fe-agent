import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from agent_core.coding.model import (
    ReadOnlyCodingResult,
)
from agent_core.coding.orchestration_model import TaskAnalysisRecord
from agent_core.context.model import (
    TaskContextBundle,
    RenderedContext,
)
from agent_core.context.renderer import (
    ContextRenderer,
)


class AnalysisArtifact(BaseModel):
    schema_version: Literal[1] = 1

    artifact_id: str

    task_id: str

    context_hash: str | None = None

    status: Literal["proposed"] = "proposed"

    result: ReadOnlyCodingResult


def calculate_context_hash(
    bundle: TaskContextBundle,
) -> str:
    # 与实际送给模型的 Context 使用同一个 Renderer。
    rendered = ContextRenderer().render(bundle)

    serialized = json.dumps(
        {
            "system_message": rendered.system_message,
            "user_message": rendered.user_message,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def calculate_messages_hash(
    messages: RenderedContext,
) -> str:
    serialized = json.dumps(
        {
            "system_message": messages.system_message,
            "user_message": messages.user_message,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_json_atomically(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:

            temporary_path = Path(file.name)

            file.write(content)
            file.flush()

            os.fsync(file.fileno())

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


class AnalysisArtifactStore:

    def __init__(
        self,
        root: str | Path,
    ) -> None:

        self.root = Path(root).resolve()

        self.artifact_dir = self.root / ".fe-agent" / "runtime" / "artifacts"

    def publish(
        self,
        *,
        bundle: TaskContextBundle,
        record: TaskAnalysisRecord,
        context_hash: str | None = None,
    ) -> tuple[AnalysisArtifact, Path]:

        if record.status != "succeeded":
            raise ValueError("只有成功的 Analysis 才能发布 Artifact")

        if record.result is None:
            raise ValueError("成功记录缺少分析结果")

        if record.task_id != bundle.task.task_id:
            raise ValueError("Record 与 Context 的 Task ID 不一致")

        if record.result.task_id != record.task_id:
            raise ValueError("分析结果的 Task ID 不一致")

        if record.result.execution_status != "not_executed":
            raise ValueError("Analysis Artifact 不允许声明已执行")

        if context_hash is None:
            context_hash = calculate_context_hash(bundle)

        if len(context_hash) != 64 or any(
            char not in "0123456789abcdef" for char in context_hash
        ):
            raise ValueError("非法 Context Hash")

        serialized_result = record.result.model_dump(mode="json")

        identity_payload = json.dumps(
            {
                "task_id": record.task_id,
                "context_hash": context_hash,
                "result": serialized_result,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        artifact_id = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()

        artifact = AnalysisArtifact(
            artifact_id=artifact_id,
            task_id=record.task_id,
            context_hash=context_hash,
            result=record.result,
        )

        # 用哈希生成文件名，而不是直接信任模型提供的路径。
        path = self.artifact_dir / f"{artifact_id}.json"

        write_json_atomically(
            path,
            artifact.model_dump_json(indent=2),
        )

        return artifact, path

    def load(
        self,
        artifact_id: str,
    ) -> AnalysisArtifact:

        if len(artifact_id) != 64 or any(
            char not in "0123456789abcdef" for char in artifact_id
        ):
            raise ValueError("非法 Artifact ID")

        path = self.artifact_dir / f"{artifact_id}.json"

        artifact = AnalysisArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )

        if artifact.artifact_id != artifact_id:
            raise ValueError("Artifact ID 与文件内容不一致")

        if artifact.result.task_id != artifact.task_id:
            raise ValueError("Artifact 中的 Task ID 不一致")

        return artifact
