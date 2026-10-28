import hashlib
import os
import stat
import tempfile
from pathlib import Path, PurePosixPath

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
from agent_core.coding.recovery_model import (
    FileRecoveryAssessment,
    RecoveryAssessment,
)


class RecoveryError(ValueError):
    """
    Workflow Recovery 无法安全执行。
    """


class RecoveryInspector:
    """
    只读恢复检查器。

    它不会修改任何文件。

    核心职责：

    Proposal
        +
    Execution Journal
        +
    Current Repository

        ↓

    判断当前执行状态是否仍然一致，
    以及能否安全 Resume / Rollback。
    """

    def __init__(
        self,
        repository_root: str | Path,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise RecoveryError("repository_root 不是目录")

    @staticmethod
    def _sha256(
        data: bytes,
    ) -> str:

        return hashlib.sha256(data).hexdigest()

    def _resolve_target(
        self,
        relative_path: str,
    ) -> Path:
        """
        安全解析仓库文件。

        当前策略：
        - 只接受 POSIX 风格相对路径
        - 禁止 ..
        - 禁止绝对路径
        - 禁止 Windows Drive Path
        - 禁止符号链接
        - 最终文件必须位于仓库内部
        """

        if (
            not relative_path
            or "\\" in relative_path
            or ":" in relative_path
            or "\x00" in relative_path
        ):
            raise RecoveryError(f"非法路径: {relative_path}")

        parts = relative_path.split("/")

        if any(part in {"", ".", ".."} for part in parts):
            raise RecoveryError(f"不允许路径穿越或非规范路径: " f"{relative_path}")

        relative = PurePosixPath(relative_path)

        if relative.is_absolute():
            raise RecoveryError("不允许绝对路径")

        current = self.root

        for part in parts:

            current = current / part

            if current.is_symlink():
                raise RecoveryError("Recovery 不允许处理符号链接: " f"{relative_path}")

        try:
            resolved = current.resolve(strict=True)

        except (
            OSError,
            RuntimeError,
        ) as exc:

            raise RecoveryError(
                "恢复目标不存在或无法解析: " f"{relative_path}"
            ) from exc

        if not resolved.is_relative_to(self.root):
            raise RecoveryError("恢复目标位于仓库范围之外: " f"{relative_path}")

        if not resolved.is_file():
            raise RecoveryError("恢复目标不是普通文件: " f"{relative_path}")

        return resolved

    def inspect(
        self,
        *,
        proposal: FilePatchProposal,
        journal: PatchExecutionJournal,
    ) -> RecoveryAssessment:
        """
        对 Proposal、Journal 和当前磁盘进行三方对账。
        """

        if proposal.task_id != journal.task_id:
            raise RecoveryError("Proposal 与 Journal " "的 Task ID 不一致")

        proposal_paths = [patch.relative_path for patch in proposal.patches]

        journal_paths = [item.relative_path for item in journal.files]

        if len(proposal_paths) != len(set(proposal_paths)):
            raise RecoveryError("Proposal 存在重复文件")

        if len(journal_paths) != len(set(journal_paths)):
            raise RecoveryError("Journal 存在重复文件")

        if set(proposal_paths) != set(journal_paths):
            raise RecoveryError("Proposal 与 Journal " "的文件集合不一致")

        patch_map = {patch.relative_path: patch for patch in proposal.patches}

        assessments: list[FileRecoveryAssessment] = []

        for record in journal.files:

            patch = patch_map[record.relative_path]

            if record.action != patch.action:
                raise RecoveryError(
                    "Journal Action " "与 Proposal 不一致: " f"{record.relative_path}"
                )

            if record.expected_sha256 != patch.expected_sha256:
                raise RecoveryError(
                    "Journal Before Hash "
                    "与 Proposal 不一致: "
                    f"{record.relative_path}"
                )

            if record.after_sha256 != patch.after_sha256:
                raise RecoveryError(
                    "Journal After Hash "
                    "与 Proposal 不一致: "
                    f"{record.relative_path}"
                )

            path = self._resolve_path(
                record.relative_path,
                allow_missing=True,
            )

            if not path.exists():

                current_state = "absent"

                current_hash = None

            else:

                try:
                    current_bytes = path.read_bytes()

                except OSError as exc:
                    raise RecoveryError(
                        "无法读取恢复目标: " f"{record.relative_path}"
                    ) from exc

                current_hash = self._sha256(current_bytes)

                if current_hash == patch.after_sha256:
                    current_state = "after"

                elif (
                    patch.action == "modify"
                    and patch.expected_sha256 is not None
                    and current_hash == patch.expected_sha256
                ):
                    current_state = "before"

                else:
                    current_state = "diverged"

            assessments.append(
                FileRecoveryAssessment(
                    relative_path=(record.relative_path),
                    action=(patch.action),
                    journal_status=(record.status),
                    current_state=(current_state),
                    current_sha256=(current_hash),
                )
            )

        (
            can_resume,
            can_rollback,
            requires_manual_recovery,
            reason,
        ) = self._decide(
            journal=journal,
            files=assessments,
        )

        return RecoveryAssessment(
            proposal_id=(journal.proposal_id),
            task_id=(journal.task_id),
            journal_status=(journal.status),
            can_resume=(can_resume),
            can_rollback=(can_rollback),
            requires_manual_recovery=(requires_manual_recovery),
            files=assessments,
            reason=reason,
        )

    @staticmethod
    def _expected_before_state(
        item: FileRecoveryAssessment,
    ) -> str:

        if item.action == "create":
            return "absent"

        return "before"

    def _decide(
        self,
        *,
        journal: PatchExecutionJournal,
        files: list[FileRecoveryAssessment],
    ) -> tuple[
        bool,
        bool,
        bool,
        str,
    ]:
        """
        返回：

        can_resume
        can_rollback
        requires_manual_recovery
        reason
        """

        # 最危险情况：
        # 磁盘上出现既不是 Before
        # 也不是 After 的版本。
        if any(item.current_state == "diverged" for item in files):
            return (
                False,
                False,
                True,
                ("存在已经发生外部修改的文件，" "禁止自动 Resume 或 Rollback"),
            )

        # ---------------------------------
        # 正常完整 Apply
        # ---------------------------------

        if journal.status == "applied":

            if all(item.current_state == "after" for item in files):
                return (
                    False,
                    True,
                    False,
                    (
                        "所有文件都处于批准后的版本；"
                        "无需 Resume，如后续验证失败可安全 Rollback"
                    ),
                )

            return (
                False,
                False,
                True,
                ("Journal 声明 applied，" "但磁盘没有全部处于 after 状态"),
            )

        # ---------------------------------
        # 执行过程中断 / 部分执行
        # ---------------------------------

        if journal.status in {
            "partially_applied",
            "applying",
        }:

            consistent = all(
                (item.journal_status == "applied" and item.current_state == "after")
                or (
                    item.journal_status == "pending"
                    and item.current_state == self._expected_before_state(item)
                )
                for item in files
            )

            if consistent:
                return (
                    True,
                    True,
                    False,
                    (
                        "磁盘状态和 Journal 完全一致；"
                        "可以继续 Pending 文件，"
                        "也可以回滚已 Applied 文件"
                    ),
                )

            return (
                False,
                False,
                True,
                ("Journal 的文件状态与" "当前磁盘状态不一致"),
            )

        # ---------------------------------
        # 尚未真正 Commit
        # ---------------------------------

        if journal.status == "staged":

            if all(
                item.journal_status == "pending"
                and item.current_state == self._expected_before_state(item)
                for item in files
            ):
                return (
                    True,
                    False,
                    False,
                    (
                        "所有文件仍为执行前版本，"
                        "尚未产生正式文件副作用，"
                        "可以重新 Stage 并继续"
                    ),
                )

            return (
                False,
                False,
                True,
                "staged 状态与磁盘不一致",
            )

        # ---------------------------------
        # Apply 在首个副作用前失败
        # ---------------------------------

        if journal.status == "failed":

            if all(
                item.current_state == self._expected_before_state(item)
                for item in files
            ):
                return (
                    False,
                    False,
                    False,
                    ("执行失败但没有留下文件副作用；" "当前无需恢复"),
                )

            return (
                False,
                False,
                True,
                ("Journal 声明 failed，" "但磁盘已经存在修改"),
            )

        # ---------------------------------
        # Rollback 已经开始
        # ---------------------------------

        if journal.status == "rolling_back":

            # 对 rolling_back：
            #
            # rolled_back → 应为 before
            # applied     → 应为 after
            # pending     → 应为 before

            consistent = all(
                (
                    item.journal_status == "rolled_back"
                    and item.current_state == self._expected_before_state(item)
                )
                or (item.journal_status == "applied" and item.current_state == "after")
                or (
                    item.journal_status == "pending"
                    and item.current_state == self._expected_before_state(item)
                )
                for item in files
            )

            if consistent:
                return (
                    False,
                    True,
                    False,
                    (
                        "Rollback 中断，"
                        "当前磁盘与 Journal 一致，"
                        "可以继续回滚剩余 applied 文件"
                    ),
                )

            return (
                False,
                False,
                True,
                ("Rollback 中断后 Journal " "与磁盘状态不一致"),
            )

        # ---------------------------------
        # Rollback 已完成
        # ---------------------------------

        if journal.status == "rolled_back":

            if all(
                item.current_state == self._expected_before_state(item)
                for item in files
            ):
                return (
                    False,
                    False,
                    False,
                    ("所有文件均已恢复为执行前版本"),
                )

            return (
                False,
                False,
                True,
                ("Journal 声明 rolled_back，" "但磁盘仍存在非 before 文件"),
            )

        # ---------------------------------
        # Rollback 自身失败
        # ---------------------------------

        if journal.status == "rollback_failed":
            # 这里不尝试自动猜测如何处理，
            # 即使没有 diverged，也建议人工确认。
            return (
                False,
                False,
                True,
                ("Rollback 曾发生失败，" "需要人工确认执行日志和仓库状态"),
            )

        return (
            False,
            False,
            True,
            ("当前 Execution Journal 状态" "无法安全自动恢复"),
        )

    def _resolve_path(
        self,
        relative_path: str,
        *,
        allow_missing: bool,
    ) -> Path:

        if (
            not relative_path
            or "\\" in relative_path
            or ":" in relative_path
            or "\x00" in relative_path
        ):
            raise RecoveryError(f"非法路径: " f"{relative_path}")

        parts = relative_path.split("/")

        if any(part in {"", ".", ".."} for part in parts):
            raise RecoveryError("不允许路径穿越或" "非规范路径: " f"{relative_path}")

        relative = PurePosixPath(relative_path)

        if relative.is_absolute():
            raise RecoveryError("不允许绝对路径")

        current = self.root

        for part in parts[:-1]:

            current = current / part

            if current.is_symlink():
                raise RecoveryError(
                    "Recovery " "不允许经过符号链接: " f"{relative_path}"
                )

        try:
            parent = current.resolve(strict=True)

        except OSError as exc:
            raise RecoveryError("恢复目标父目录" "不存在: " f"{relative_path}") from exc

        if not parent.is_relative_to(self.root):
            raise RecoveryError("恢复目标位于仓库之外")

        target = parent / parts[-1]

        if target.is_symlink():
            raise RecoveryError("Recovery " "不允许处理符号链接: " f"{relative_path}")

        if target.exists():

            if not target.is_file():
                raise RecoveryError("恢复目标不是普通文件: " f"{relative_path}")

            return target

        if not allow_missing:
            raise RecoveryError("恢复目标不存在: " f"{relative_path}")

        return target


class SafeRollbackService:
    """
    安全回滚服务。

    核心规则：

    只有当前文件仍然等于 Agent 写入后的
    after_sha256 时，才允许恢复成 before_content。

    如果当前文件已经被其他人修改，
    禁止自动覆盖。
    """

    def __init__(
        self,
        *,
        repository_root: str | Path,
        journal_store: ExecutionJournalStore,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise RecoveryError("repository_root 不是目录")

        self.inspector = RecoveryInspector(self.root)

        self.journal_store = journal_store

    @staticmethod
    def _sha256(
        data: bytes,
    ) -> str:

        return hashlib.sha256(data).hexdigest()

    def _rollback_file(
        self,
        *,
        patch: ValidatedFilePatch,
    ) -> None:

        path = self.inspector._resolve_path(
            patch.relative_path,
            allow_missing=False,
        )

        try:
            current_bytes = path.read_bytes()

        except OSError as exc:
            raise RecoveryError(
                "无法读取待回滚文件: " f"{patch.relative_path}"
            ) from exc

        if self._sha256(current_bytes) != patch.after_sha256:
            raise RecoveryError(
                "文件已经不是 Agent "
                "写入后的版本，"
                "禁止自动回滚: "
                f"{patch.relative_path}"
            )

        # ==================================
        # CREATE rollback
        #
        # Agent 创建的文件：
        #
        # after
        #   ↓
        # absent
        #
        # 只有文件仍然严格等于
        # Agent after 版本时才允许删除。
        # ==================================

        if patch.action == "create":

            # 删除前再次校验，
            # 防止检查和删除之间被外部修改。
            current_bytes = path.read_bytes()

            if self._sha256(current_bytes) != patch.after_sha256:
                raise RecoveryError(
                    "删除 create target 前" "文件发生变化: " f"{patch.relative_path}"
                )

            path.unlink()

            if path.exists():
                raise RecoveryError(
                    "create target " "Rollback 删除失败: " f"{patch.relative_path}"
                )

            return

        # ==================================
        # MODIFY rollback
        # ==================================

        if patch.before_content is None:
            raise RecoveryError("modify patch " "缺少 before_content")

        before_bytes = patch.before_content.encode("utf-8")

        temporary_path: Path | None = None

        try:
            original_mode = stat.S_IMODE(path.stat().st_mode)

            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=(f".{path.name}."),
                suffix=(".rollback.tmp"),
                delete=False,
            ) as temporary_file:

                temporary_path = Path(temporary_file.name)

                temporary_file.write(before_bytes)

                temporary_file.flush()

                os.fsync(temporary_file.fileno())

            os.chmod(
                temporary_path,
                original_mode,
            )

            current_bytes = path.read_bytes()

            if self._sha256(current_bytes) != patch.after_sha256:
                raise RecoveryError(
                    "回滚提交前" "文件发生变化: " f"{patch.relative_path}"
                )

            os.replace(
                temporary_path,
                path,
            )

            temporary_path = None

            restored = path.read_bytes()

            expected_before_hash = self._sha256(before_bytes)

            if self._sha256(restored) != expected_before_hash:
                raise RecoveryError(
                    "Rollback 写入后 " "Hash 校验失败: " f"{patch.relative_path}"
                )

        finally:

            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def rollback(
        self,
        *,
        proposal: FilePatchProposal,
        journal: PatchExecutionJournal,
    ) -> RecoveryAssessment:
        """
        回滚所有当前仍处于 Agent after 版本的文件。

        注意：
        pending 文件从未被 Agent 修改，
        不需要回滚。
        """

        assessment = self.inspector.inspect(
            proposal=proposal,
            journal=journal,
        )

        if not assessment.can_rollback:
            raise RecoveryError("当前状态不允许自动 Rollback: " + assessment.reason)

        patch_map = {patch.relative_path: patch for patch in proposal.patches}

        rollback_targets = [
            item for item in assessment.files if (item.current_state == "after")
        ]

        # 如果没有 after 文件，
        # 实际已经没有东西需要回滚。
        if not rollback_targets:
            journal.status = "rolled_back"

            journal.error = None

            self.journal_store.save(journal)

            return self.inspector.inspect(
                proposal=proposal,
                journal=journal,
            )

        # ---------------------------------
        # Precheck ALL
        # ---------------------------------

        # 在修改任何文件之前，
        # 先确认所有 rollback target
        # 都仍然处于 after 版本。
        for item in rollback_targets:

            patch = patch_map[item.relative_path]

            path = self.inspector._resolve_target(item.relative_path)

            current_hash = self._sha256(path.read_bytes())

            if current_hash != patch.after_sha256:
                raise RecoveryError("Rollback 前版本检查失败: " f"{item.relative_path}")

        # ---------------------------------
        # Mark rolling_back
        # ---------------------------------

        journal.status = "rolling_back"

        journal.error = None

        self.journal_store.save(journal)

        try:

            for item in rollback_targets:

                patch = patch_map[item.relative_path]

                self._rollback_file(patch=patch)

                # 每产生一次真实恢复副作用，
                # 立即更新 Journal。
                for record in journal.files:

                    if record.relative_path == item.relative_path:
                        record.status = "rolled_back"
                        break

                self.journal_store.save(journal)

            journal.status = "rolled_back"

            journal.error = None

            self.journal_store.save(journal)

        except Exception as exc:

            journal.status = "rollback_failed"

            journal.error = f"{type(exc).__name__}: " f"{exc}"

            self.journal_store.save(journal)

            raise

        return self.inspector.inspect(
            proposal=proposal,
            journal=journal,
        )
