import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Protocol

from agent_core.coding.handoff import (
    AnalysisArtifact,
    AnalysisArtifactStore,
    calculate_messages_hash,
)
from agent_core.coding.preparation_model import (
    ExecutionPreparation,
)
from agent_core.context.model import (
    CodeContextItem,
    RenderedContext,
    TaskContextBundle,
)


class PreparationError(ValueError):
    """执行前校验未通过。"""


class PreparedMessageProvider(Protocol):

    def prepare_messages(
        self,
        bundle: TaskContextBundle,
    ) -> RenderedContext: ...


class ExecutionPreparer:

    def __init__(
        self,
        *,
        repository_root: str | Path,
        artifact_store: AnalysisArtifactStore,
        message_provider: PreparedMessageProvider,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise PreparationError("仓库根路径不是目录")

        self.artifact_store = artifact_store

        self.message_provider = message_provider

    def prepare(
        self,
        *,
        task_id: str,
        artifact_id: str,
        bundle: TaskContextBundle,
    ) -> ExecutionPreparation:

        # 1. 读取并校验 Artifact。
        artifact = self.artifact_store.load(artifact_id)

        self._validate_artifact_integrity(artifact)

        # 2. 校验当前 Task。
        if artifact.task_id != task_id:
            raise PreparationError("Artifact 不属于当前 Task")

        if bundle.task.task_id != task_id:
            raise PreparationError("Context 不属于当前 Task")

        if artifact.result.execution_status != "not_executed":
            raise PreparationError("分析结果不能声明已经执行")

        # 3. 用实际消息格式重新计算 Hash。
        messages = self.message_provider.prepare_messages(bundle)

        actual_context_hash = calculate_messages_hash(messages)

        if actual_context_hash != artifact.context_hash:
            raise PreparationError("Context Hash 不一致：" "当前上下文与分析时不同")

        # 4. 检查 Context 所引用的实际文件。
        context_paths: set[str] = set()

        for item in bundle.code_context:

            normalized = self._normalize_path(item.relative_path)

            if normalized in context_paths:
                raise PreparationError(f"重复的 Code Context 文件: {normalized}")

            context_paths.add(normalized)

            self._validate_code_freshness(item)

        # 5. 检查模型提出的修改目标。
        proposed_paths: list[str] = []

        for change in artifact.result.proposed_changes:

            normalized = self._normalize_path(change.relative_path)

            if normalized not in context_paths:
                raise PreparationError(
                    "修改目标不在已验证的 Context 中: " f"{normalized}"
                )

            proposed_paths.append(normalized)

        return ExecutionPreparation(
            task_id=task_id,
            artifact_id=(artifact.artifact_id),
            context_hash=(artifact.context_hash),
            proposed_paths=sorted(set(proposed_paths)),
            checked_context_paths=sorted(context_paths),
            checks_passed=[
                "artifact_integrity",
                "task_consistency",
                "context_consistency",
                "repository_freshness",
                "path_boundary",
            ],
        )

    def _validate_artifact_integrity(
        self,
        artifact: AnalysisArtifact,
    ) -> None:

        # 与 ArtifactStore.publish() 使用相同的
        # ID 计算规则。
        identity_payload = json.dumps(
            {
                "task_id": artifact.task_id,
                "context_hash": (artifact.context_hash),
                "result": (artifact.result.model_dump(mode="json")),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        expected_id = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()

        if expected_id != artifact.artifact_id:
            raise PreparationError("Artifact 内容完整性校验失败")

    def _normalize_path(
        self,
        relative_path: str,
    ) -> str:

        if not relative_path:
            raise PreparationError("文件路径不能为空")

        # 统一只接受 Repository Index 使用的
        # POSIX 风格相对路径。
        if "\\" in relative_path:
            raise PreparationError("文件路径不能包含反斜杠")

        if "\x00" in relative_path or ":" in relative_path:
            raise PreparationError("文件路径包含非法字符")

        if re.match(
            r"^[A-Za-z]:",
            relative_path,
        ):
            raise PreparationError("不允许 Windows 绝对路径")

        path = PurePosixPath(relative_path)

        if path.is_absolute():
            raise PreparationError("不允许绝对路径")

        if any(part in {"", ".", ".."} for part in relative_path.split("/")):
            raise PreparationError("不允许路径穿越或非规范路径")

        return path.as_posix()

    def _resolve_repository_file(
        self,
        relative_path: str,
    ) -> Path:

        normalized = self._normalize_path(relative_path)

        try:
            resolved = (self.root / normalized).resolve(strict=True)

        except (
            OSError,
            RuntimeError,
        ) as exc:
            raise PreparationError(f"无法解析文件路径: {normalized}") from exc

        if not resolved.is_relative_to(self.root):
            raise PreparationError("文件解析后位于仓库范围之外: " f"{normalized}")

        if not resolved.is_file():
            raise PreparationError(f"目标不是普通文件: {normalized}")

        return resolved

    def _validate_code_freshness(
        self,
        item: CodeContextItem,
    ) -> None:

        path = self._resolve_repository_file(item.relative_path)

        try:
            current_content = path.read_text(encoding="utf-8")

        except (
            UnicodeDecodeError,
            OSError,
        ) as exc:
            raise PreparationError(
                "无法读取当前代码文件: " f"{item.relative_path}"
            ) from exc

        if item.content_mode == "full_file":

            if current_content != item.content:
                raise PreparationError("代码文件已经发生变化: " f"{item.relative_path}")

            return

        if item.content_mode == "truncated":

            if not current_content.startswith(item.content):
                raise PreparationError(
                    "截断上下文对应的代码已变化: " f"{item.relative_path}"
                )

            return

        if item.content_mode == "semantic_chunk":

            if item.start_line is None or item.end_line is None:
                raise PreparationError("Semantic Chunk 缺少行号")

            if item.end_line < item.start_line:
                raise PreparationError("Semantic Chunk 行号非法")

            lines = current_content.splitlines()

            current_chunk = "\n".join(lines[item.start_line - 1 : item.end_line])

            if not current_chunk.startswith(item.content):
                raise PreparationError(
                    "Semantic Chunk 已过期: " f"{item.relative_path}"
                )

            return

        raise PreparationError(f"未知的内容模式: {item.content_mode}")
