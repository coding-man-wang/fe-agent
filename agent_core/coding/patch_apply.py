import hashlib
import os
import stat
import tempfile
from pathlib import Path, PurePosixPath

from agent_core.coding.approval_store import (
    ApprovalStore,
)
from agent_core.coding.patch_apply_model import (
    PatchExecutionResult,
)
from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
    ValidatedFilePatch,
)
from agent_core.coding.reviewer import (
    PatchReviewer,
)


class PatchApplyError(ValueError):
    """Patch 应用失败。"""


class PatchApplyService:

    def __init__(
        self,
        *,
        repository_root: str | Path,
        approval_store: ApprovalStore,
    ) -> None:
        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise PatchApplyError("仓库根目录不是目录")

        self.approval_store = approval_store

        self.reviewer = PatchReviewer()

    def _resolve_target(
        self,
        relative_path: str,
    ) -> Path:

        if (
            not relative_path
            or "\\" in relative_path
            or ":" in relative_path
            or "\x00" in relative_path
        ):
            raise PatchApplyError("非法文件路径")

        parts = relative_path.split("/")

        if any(part in {"", ".", ".."} for part in parts):
            raise PatchApplyError("不允许路径穿越或非规范路径")

        relative = PurePosixPath(relative_path)

        if relative.is_absolute():
            raise PatchApplyError("不允许绝对路径")

        current = self.root

        for part in parts:

            current = current / part

            if current.is_symlink():
                raise PatchApplyError("不允许通过符号链接写入文件")

        try:
            resolved = current.resolve(strict=True)

        except (
            OSError,
            RuntimeError,
        ) as exc:
            raise PatchApplyError("无法解析目标文件") from exc

        if not resolved.is_relative_to(self.root):
            raise PatchApplyError("目标文件位于仓库外")

        if not resolved.is_file():
            raise PatchApplyError("当前只允许修改已有普通文件")

        return resolved

    @staticmethod
    def _sha256(
        data: bytes,
    ) -> str:

        return hashlib.sha256(data).hexdigest()

    def _validate_original(
        self,
        *,
        path: Path,
        patch: ValidatedFilePatch,
    ) -> bytes:

        data = path.read_bytes()

        actual_hash = self._sha256(data)

        if actual_hash != patch.expected_sha256:
            raise PatchApplyError("目标文件版本已经变化，拒绝写入")

        expected_content = patch.before_content.encode("utf-8")

        if data != expected_content:
            raise PatchApplyError("修改前内容与 Proposal 不一致")

        return data

    def _replace_atomically(
        self,
        *,
        path: Path,
        before_bytes: bytes,
        after_bytes: bytes,
    ) -> None:

        temporary_path: Path | None = None

        original_mode = stat.S_IMODE(path.stat().st_mode)

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".fe-agent.tmp",
                delete=False,
            ) as temporary_file:

                temporary_path = Path(temporary_file.name)

                temporary_file.write(after_bytes)

                temporary_file.flush()

                os.fsync(temporary_file.fileno())

            # 尽量保留目标文件原有的权限位。
            os.chmod(
                temporary_path,
                original_mode,
            )

            # 提交前再验证一次，减少检查与写入的时间差。
            current_path = self._resolve_target(path.relative_to(self.root).as_posix())

            current_bytes = current_path.read_bytes()

            if current_bytes != before_bytes:
                raise PatchApplyError("提交前文件发生变化，拒绝替换")

            os.replace(
                temporary_path,
                current_path,
            )

            temporary_path = None

        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def apply(
        self,
        *,
        proposal: FilePatchProposal,
    ) -> PatchExecutionResult:

        # 1. 重新构建 ReviewPackage，
        #    检查 Patch 内容、Hash 和 Diff。
        review = self.reviewer.prepare(proposal)

        # 2. 从持久化存储中读取真实审批记录。
        self.approval_store.require_approved(review=review)

        # 3. v0.1 明确只支持一个文件。
        if len(proposal.patches) != 1:
            raise PatchApplyError("v0.1 只支持单文件 Patch Apply")

        patch = proposal.patches[0]

        # 4. 安全解析路径。
        path = self._resolve_target(patch.relative_path)

        # 5. 重新检查原始文件版本。
        before_bytes = self._validate_original(
            path=path,
            patch=patch,
        )

        after_bytes = patch.after_content.encode("utf-8")

        # 6. 原子替换。
        self._replace_atomically(
            path=path,
            before_bytes=before_bytes,
            after_bytes=after_bytes,
        )

        # 7. 写入后校验。
        actual_after = path.read_bytes()

        if self._sha256(actual_after) != patch.after_sha256:
            raise PatchApplyError(
                "文件已写入，但结果 Hash 校验失败；" "需要人工检查当前文件状态"
            )

        return PatchExecutionResult(
            task_id=proposal.task_id,
            proposal_id=review.proposal_id,
            relative_path=patch.relative_path,
            before_sha256=(patch.expected_sha256),
            after_sha256=(patch.after_sha256),
            status="file_written",
            verification_status="not_run",
        )
