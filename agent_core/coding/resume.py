import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agent_core.coding.approval_store import (
    ApprovalStore,
)
from agent_core.coding.execution_journal import (
    ExecutionJournalStore,
)
from agent_core.coding.execution_journal_model import (
    PatchExecutionJournal,
)
from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
    ValidatedFilePatch,
)
from agent_core.coding.recovery import (
    RecoveryInspector,
)
from agent_core.coding.reviewer import (
    PatchReviewer,
)


class ResumeError(ValueError):
    """
    Execution Resume 无法安全继续。
    """


@dataclass
class StagedResumeFile:
    """
    Resume 期间的进程内临时数据。

    不需要持久化。
    """

    patch: ValidatedFilePatch

    target_path: Path

    temporary_path: Path

    before_bytes: bytes


class PatchResumeService:
    """
    针对 partially_applied / applying / staged
    Execution Journal 继续执行尚未完成的文件。
    """

    def __init__(
        self,
        *,
        repository_root: str | Path,
        approval_store: ApprovalStore,
        journal_store: ExecutionJournalStore,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise ResumeError("repository_root 不是目录")

        self.approval_store = approval_store

        self.journal_store = journal_store

        self.reviewer = PatchReviewer()

        self.inspector = RecoveryInspector(self.root)

    @staticmethod
    def _sha256(
        data: bytes,
    ) -> str:

        return hashlib.sha256(data).hexdigest()

    def _stage_pending_file(
        self,
        patch: ValidatedFilePatch,
    ) -> StagedResumeFile:
        """
        Resume 时只 Stage 尚未执行的文件。

        当前文件必须仍然是 Proposal 的 BEFORE 版本。
        """

        path = self.inspector._resolve_target(patch.relative_path)

        try:
            before_bytes = path.read_bytes()

        except OSError as exc:
            raise ResumeError(
                "无法读取 Pending 文件: " f"{patch.relative_path}"
            ) from exc

        if self._sha256(before_bytes) != patch.expected_sha256:
            raise ResumeError("Pending 文件已经变化: " f"{patch.relative_path}")

        if before_bytes != patch.before_content.encode("utf-8"):
            raise ResumeError(
                "Pending 文件内容与 Proposal 不一致: " f"{patch.relative_path}"
            )

        after_bytes = patch.after_content.encode("utf-8")

        if self._sha256(after_bytes) != patch.after_sha256:
            raise ResumeError("Proposal After Hash 不一致: " f"{patch.relative_path}")

        temporary_path: Path | None = None

        try:
            original_mode = stat.S_IMODE(path.stat().st_mode)

            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".resume.stage",
                delete=False,
            ) as file:

                temporary_path = Path(file.name)

                file.write(after_bytes)

                file.flush()

                os.fsync(file.fileno())

            os.chmod(
                temporary_path,
                original_mode,
            )

        except Exception:

            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

            raise

        return StagedResumeFile(
            patch=patch,
            target_path=path,
            temporary_path=temporary_path,
            before_bytes=before_bytes,
        )

    @staticmethod
    def _cleanup(
        staged: list[StagedResumeFile],
    ) -> None:

        for item in staged:

            if item.temporary_path.exists():
                try:
                    item.temporary_path.unlink()

                except OSError:
                    pass

    def _commit(
        self,
        staged: StagedResumeFile,
    ) -> None:

        patch = staged.patch

        try:
            current_bytes = staged.target_path.read_bytes()

        except OSError as exc:
            raise ResumeError(
                "Commit 前无法读取文件: " f"{patch.relative_path}"
            ) from exc

        # Resume Commit 前再次检查，
        # 防止 Stage 后文件被其他程序修改。
        if current_bytes != staged.before_bytes:
            raise ResumeError("Resume Commit 前文件发生变化: " f"{patch.relative_path}")

        os.replace(
            staged.temporary_path,
            staged.target_path,
        )

        try:
            current_after = staged.target_path.read_bytes()

        except OSError as exc:
            raise ResumeError(
                "Resume 写入后无法读取文件: " f"{patch.relative_path}"
            ) from exc

        if self._sha256(current_after) != patch.after_sha256:
            raise ResumeError("Resume 写入后的 Hash 不一致: " f"{patch.relative_path}")

    def resume(
        self,
        *,
        proposal: FilePatchProposal,
        journal: PatchExecutionJournal,
    ) -> PatchExecutionJournal:
        """
        继续执行 Pending Patch。

        已经 applied 的文件绝不会重复写入。
        """

        # ---------------------------------
        # 1. Proposal Integrity
        # ---------------------------------

        review = self.reviewer.prepare(proposal)

        if review.proposal_id != journal.proposal_id:
            raise ResumeError("Journal 不属于当前 Proposal")

        # ---------------------------------
        # 2. Approval 仍然有效
        # ---------------------------------

        self.approval_store.require_approved(review=review)

        # ---------------------------------
        # 3. Recovery Assessment
        # ---------------------------------

        assessment = self.inspector.inspect(
            proposal=proposal,
            journal=journal,
        )

        if not assessment.can_resume:
            raise ResumeError("当前状态不允许 Resume: " + assessment.reason)

        patch_map = {patch.relative_path: patch for patch in proposal.patches}

        pending_paths = {
            item.relative_path for item in journal.files if item.status == "pending"
        }

        # ---------------------------------
        # 4. 如果实际上没有 Pending
        # ---------------------------------

        if not pending_paths:
            journal.status = "applied"

            journal.error = None

            self.journal_store.save(journal)

            return journal

        # ---------------------------------
        # 5. Stage ALL Pending Files
        # ---------------------------------

        staged: list[StagedResumeFile] = []

        try:

            for relative_path in sorted(pending_paths):

                patch = patch_map.get(relative_path)

                if patch is None:
                    raise ResumeError(
                        "Pending 文件不存在于 Proposal: " f"{relative_path}"
                    )

                staged.append(self._stage_pending_file(patch))

        except Exception:

            self._cleanup(staged)

            raise

        # ---------------------------------
        # 6. 标记重新进入 applying
        # ---------------------------------

        journal.status = "applying"

        journal.error = None

        self.journal_store.save(journal)

        # ---------------------------------
        # 7. Commit Pending Files
        # ---------------------------------

        try:

            for staged_file in staged:

                self._commit(staged_file)

                for record in journal.files:

                    if record.relative_path == staged_file.patch.relative_path:
                        record.status = "applied"
                        break

                # 每产生一次真实副作用，
                # 立即记录。
                self.journal_store.save(journal)

            journal.status = "applied"

            journal.error = None

            self.journal_store.save(journal)

            return journal

        except Exception as exc:

            applied_count = sum(1 for item in journal.files if item.status == "applied")

            pending_count = sum(1 for item in journal.files if item.status == "pending")

            if applied_count > 0 and pending_count > 0:
                journal.status = "partially_applied"

            elif pending_count == 0:
                # 极少数情况下：
                # 文件已全部写完，但最后状态操作报错。
                journal.status = "applying"

            else:
                journal.status = "failed"

            journal.error = f"{type(exc).__name__}: " f"{exc}"

            self.journal_store.save(journal)

            raise

        finally:

            self._cleanup(staged)
