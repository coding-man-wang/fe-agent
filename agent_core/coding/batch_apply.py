import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from agent_core.coding.approval_store import ApprovalStore
from agent_core.coding.execution_journal import ExecutionJournalStore
from agent_core.coding.execution_journal_model import (
    PatchExecutionJournal,
    FileExecutionRecord,
)
from agent_core.coding.patch_generation_model import (
    ValidatedFilePatch,
    FilePatchProposal,
)
from agent_core.coding.reviewer import PatchReviewer


class BatchPatchApplyError(ValueError):
    pass


@dataclass
class StagedFile:
    patch: ValidatedFilePatch

    target_path: Path

    temporary_path: Path

    before_bytes: bytes


class BatchPatchApplyService:

    def __init__(
        self,
        *,
        repository_root: str | Path,
        approval_store: ApprovalStore,
        journal_store: ExecutionJournalStore,
    ) -> None:
        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise BatchPatchApplyError("repository_root 不是目录")

        self.approval_store = approval_store

        self.journal_store = journal_store

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
            raise BatchPatchApplyError(f"非法路径: {relative_path}")

        parts = relative_path.split("/")

        if any(part in {"", ".", ".."} for part in parts):
            raise BatchPatchApplyError("不允许路径穿越")

        relative = PurePosixPath(relative_path)

        if relative.is_absolute():
            raise BatchPatchApplyError("不允许绝对路径")

        current = self.root

        for part in parts:

            current = current / part

            if current.is_symlink():
                raise BatchPatchApplyError("不允许通过符号链接写文件")

        try:
            resolved = current.resolve(strict=True)

        except OSError as exc:
            raise BatchPatchApplyError(f"无法解析文件: {relative_path}") from exc

        if not resolved.is_relative_to(self.root):
            raise BatchPatchApplyError("目标位于仓库之外")

        if not resolved.is_file():
            raise BatchPatchApplyError("只能修改已有普通文件")

        return resolved

    @staticmethod
    def _sha256(
        data: bytes,
    ) -> str:

        return hashlib.sha256(data).hexdigest()

    def _stage_file(
        self,
        patch: ValidatedFilePatch,
    ) -> StagedFile:

        target = self._resolve_target(patch.relative_path)

        before_bytes = target.read_bytes()

        actual_hash = self._sha256(before_bytes)

        if actual_hash != (patch.expected_sha256):
            raise BatchPatchApplyError("文件版本已变化: " f"{patch.relative_path}")

        if before_bytes != (patch.before_content.encode("utf-8")):
            raise BatchPatchApplyError(
                "文件内容与 Proposal 不一致: " f"{patch.relative_path}"
            )

        after_bytes = patch.after_content.encode("utf-8")

        if self._sha256(after_bytes) != patch.after_sha256:
            raise BatchPatchApplyError("After Hash 不一致: " f"{patch.relative_path}")

        original_mode = stat.S_IMODE(target.stat().st_mode)

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".fe-agent.stage",
            delete=False,
        ) as temporary_file:

            temporary_path = Path(temporary_file.name)

            temporary_file.write(after_bytes)

            temporary_file.flush()

            os.fsync(temporary_file.fileno())

        os.chmod(
            temporary_path,
            original_mode,
        )

        return StagedFile(
            patch=patch,
            target_path=target,
            temporary_path=temporary_path,
            before_bytes=before_bytes,
        )

    def _stage_all(
        self,
        proposal: FilePatchProposal,
    ) -> list[StagedFile]:

        staged: list[StagedFile] = []

        try:
            for patch in proposal.patches:
                staged.append(self._stage_file(patch))

            return staged

        except Exception:

            self._cleanup_staged(staged)

            raise

    @staticmethod
    def _cleanup_staged(
        staged: list[StagedFile],
    ) -> None:
        """
        清理尚未提交的临时 staged 文件。

        注意：
        - 只删除 temporary_path
        - 不修改正式文件
        - 已经被 os.replace() 的临时文件通常已经不存在
        """

        for item in staged:

            if item.temporary_path.exists():

                try:
                    item.temporary_path.unlink()

                except OSError:
                    # 当前阶段清理临时文件失败不应该覆盖
                    # 原始 Apply 异常。
                    #
                    # 后续可以将这里升级成 cleanup warning
                    # 或 Execution Journal cleanup 状态。
                    pass

    def _create_journal(
        self,
        *,
        proposal: FilePatchProposal,
        proposal_id: str,
    ) -> PatchExecutionJournal:

        journal = PatchExecutionJournal(
            proposal_id=proposal_id,
            task_id=proposal.task_id,
            status="staged",
            files=[
                FileExecutionRecord(
                    relative_path=(patch.relative_path),
                    expected_sha256=(patch.expected_sha256),
                    after_sha256=(patch.after_sha256),
                )
                for patch in proposal.patches
            ],
        )

        self.journal_store.create(journal)

        return journal

    def _commit_staged_file(
        self,
        staged: StagedFile,
    ) -> None:

        patch = staged.patch

        # Commit 前再次读取正式文件。
        current = staged.target_path.read_bytes()

        if current != (staged.before_bytes):
            raise BatchPatchApplyError(
                "Commit 前文件发生变化: " f"{patch.relative_path}"
            )

        os.replace(
            staged.temporary_path,
            staged.target_path,
        )

        current_after = staged.target_path.read_bytes()

        if self._sha256(current_after) != patch.after_sha256:
            raise BatchPatchApplyError(
                "写入后的文件 Hash 不一致: " f"{patch.relative_path}"
            )

    def exists(
        self,
        proposal_id: str,
    ) -> bool:

        return self._path(proposal_id).exists()

    def apply(
        self,
        *,
        proposal: FilePatchProposal,
    ) -> PatchExecutionJournal:

        if len(proposal.patches) < 1:
            raise BatchPatchApplyError("Proposal 没有 Patch")

        # 重新验证 Proposal。
        review = self.reviewer.prepare(proposal)

        # 必须有真实持久化审批。
        self.approval_store.require_approved(review=review)

        # 防止同一 Proposal 被简单重复执行。
        if self.journal_store.exists(review.proposal_id):
            raise BatchPatchApplyError("该 Proposal 已存在 Execution Journal")

        # 先准备全部文件。
        staged = self._stage_all(proposal)

        try:
            journal = self._create_journal(
                proposal=proposal,
                proposal_id=(review.proposal_id),
            )

        except Exception:

            self._cleanup_staged(staged)

            raise

        journal.status = "applying"

        self.journal_store.save(journal)

        try:
            for staged_file in staged:

                self._commit_staged_file(staged_file)

                for record in journal.files:

                    if record.relative_path == staged_file.patch.relative_path:
                        record.status = "applied"
                        break

                # 每完成一个真实副作用，
                # 立即持久化 Journal。
                self.journal_store.save(journal)

            journal.status = "applied"

            journal.error = None

            self.journal_store.save(journal)

            return journal

        except Exception as exc:

            applied_count = sum(1 for item in journal.files if item.status == "applied")

            if applied_count > 0:

                journal.status = "partially_applied"

            else:

                journal.status = "failed"

            journal.error = f"{type(exc).__name__}: " f"{exc}"

            self.journal_store.save(journal)

            raise

        finally:
            self._cleanup_staged(staged)
