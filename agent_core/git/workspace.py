import hashlib
import json
import re
from pathlib import Path

from agent_core.git.cli_client import (
    CliGitClient,
)
from agent_core.git.client import (
    GitClient,
)
from agent_core.git.errors import (
    GitWorkspaceConflictError,
    GitWorkspaceError,
)
from agent_core.git.repository import (
    GitRepositoryInspector,
)
from agent_core.git.workspace_model import (
    GitWorkspaceRecord,
)
from agent_core.git.workspace_store import (
    GitWorkspaceStore,
)


class AgentGitWorkspaceManager:

    def __init__(
        self,
        *,
        repository_root: str | Path,
        workspace_root: str | Path,
        git_client: GitClient | None = None,
        inspector: GitRepositoryInspector | None = None,
        store: GitWorkspaceStore | None = None,
    ) -> None:

        self.repository_root = Path(repository_root).resolve()

        self.workspace_root = Path(workspace_root).resolve()

        self.git_client = git_client or CliGitClient()

        self.inspector = inspector or GitRepositoryInspector(git_client=self.git_client)

        self.store = store or GitWorkspaceStore(self.repository_root)

        self._validate_workspace_root()

    async def create(
        self,
        *,
        task_id: str,
        proposal_id: str,
        base_commit_sha: str,
    ) -> GitWorkspaceRecord:

        self._validate_hash(
            proposal_id,
            name="Proposal ID",
        )

        self._validate_hash(
            base_commit_sha,
            name="Base Commit SHA",
        )

        repository_state = await self.inspector.inspect(self.repository_root)

        actual_root = Path(repository_state.repository_root).resolve()

        if actual_root != self.repository_root:
            raise GitWorkspaceConflictError(
                "配置的 Repository Root " "与 Git 实际 Root 不一致"
            )

        await self._validate_base_commit(base_commit_sha)

        workspace_id = self.calculate_workspace_id(
            task_id=task_id,
            proposal_id=proposal_id,
            base_commit_sha=(base_commit_sha),
        )

        if self.store.exists(workspace_id):
            return await self._reuse_existing(workspace_id)

        branch_name = self.build_branch_name(
            task_id=task_id,
            workspace_id=(workspace_id),
        )

        worktree_path = self.workspace_root / workspace_id

        if worktree_path.exists():
            raise GitWorkspaceConflictError(
                "Workspace 路径已存在，" "但不存在对应持久化记录: " f"{worktree_path}"
            )

        record = GitWorkspaceRecord(
            workspace_id=workspace_id,
            task_id=task_id,
            proposal_id=proposal_id,
            repository_root=str(self.repository_root),
            base_commit_sha=(base_commit_sha),
            branch_name=branch_name,
            worktree_path=str(worktree_path),
            status="creating",
        )

        # Side Effect 前先持久化 creating。
        self.store.save(record)

        try:
            self.workspace_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            await self.git_client.run(
                cwd=self.repository_root,
                args=[
                    "worktree",
                    "add",
                    "-b",
                    branch_name,
                    str(worktree_path),
                    base_commit_sha,
                ],
            )

            await self._validate_created_workspace(record)

            ready_record = record.model_copy(
                update={
                    "status": "ready",
                    "error": None,
                }
            )

            self.store.save(ready_record)

            return ready_record

        except Exception as exc:

            failed_record = record.model_copy(
                update={
                    "status": "failed",
                    "error": (f"{type(exc).__name__}: " f"{exc}"),
                }
            )

            self.store.save(failed_record)

            raise

    async def _reuse_existing(
        self,
        workspace_id: str,
    ) -> GitWorkspaceRecord:

        record = self.store.load(workspace_id)

        if record.status == "ready":
            await self._validate_existing_workspace(record)

            return record

        if record.status == "creating":

            # 可能发生：
            #
            # git worktree add 已成功
            # 但进程在 save(ready) 前退出。
            #
            # 尝试三方对账。
            if Path(record.worktree_path).exists():
                await self._validate_created_workspace(record)

                ready = record.model_copy(
                    update={
                        "status": "ready",
                        "error": None,
                    }
                )

                self.store.save(ready)

                return ready

            raise GitWorkspaceConflictError(
                "Workspace 记录处于 creating，"
                "但 Worktree 不存在，"
                "需要显式恢复处理"
            )

        if record.status == "failed":
            raise GitWorkspaceConflictError("Workspace 创建曾失败，" "当前禁止自动重试")

        if record.status == "removed":
            raise GitWorkspaceConflictError("Workspace 已被移除")

        raise GitWorkspaceConflictError("未知 Workspace 状态: " f"{record.status}")

    async def _validate_base_commit(
        self,
        commit_sha: str,
    ) -> None:

        result = await self.git_client.run(
            cwd=self.repository_root,
            args=[
                "cat-file",
                "-e",
                (f"{commit_sha}" "^{commit}"),
            ],
            allow_failure=True,
        )

        if result.exit_code != 0:
            raise GitWorkspaceError("Base Commit 不存在: " f"{commit_sha}")

    async def _validate_created_workspace(
        self,
        record: GitWorkspaceRecord,
    ) -> None:

        path = Path(record.worktree_path)

        if not path.is_dir():
            raise GitWorkspaceConflictError("Git 命令完成后 " "Worktree 目录不存在")

        state = await self.inspector.inspect(path)

        if Path(state.repository_root).resolve() != path.resolve():
            raise GitWorkspaceConflictError("Worktree Root 不一致")

        if state.branch_name != record.branch_name:
            raise GitWorkspaceConflictError("Worktree Branch 不一致")

        if state.head_sha != record.base_commit_sha:
            raise GitWorkspaceConflictError(
                "新建 Worktree HEAD " "与 Base Commit 不一致"
            )

    async def _validate_existing_workspace(
        self,
        record: GitWorkspaceRecord,
    ) -> None:

        path = Path(record.worktree_path)

        if not path.is_dir():
            raise GitWorkspaceConflictError(
                "Workspace Record 存在，" "但 Worktree 已不存在"
            )

        state = await self.inspector.inspect(path)

        if state.branch_name != record.branch_name:
            raise GitWorkspaceConflictError("Workspace Branch " "与持久化记录不一致")

    @staticmethod
    def calculate_workspace_id(
        *,
        task_id: str,
        proposal_id: str,
        base_commit_sha: str,
    ) -> str:

        payload = json.dumps(
            {
                "task_id": task_id,
                "proposal_id": (proposal_id),
                "base_commit_sha": (base_commit_sha),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def build_branch_name(
        *,
        task_id: str,
        workspace_id: str,
    ) -> str:

        normalized_task = task_id.strip().lower()

        normalized_task = re.sub(
            r"[^a-z0-9._-]+",
            "-",
            normalized_task,
        )

        normalized_task = normalized_task.strip("-._")

        if not normalized_task:
            normalized_task = "task"

        return "fe-agent/" f"{normalized_task}-" f"{workspace_id[:12]}"

    def _validate_workspace_root(
        self,
    ) -> None:

        if (
            self.workspace_root == self.repository_root
            or self.workspace_root.is_relative_to(self.repository_root)
        ):
            raise ValueError("Workspace Root " "不能位于 Repository Root 内部")

    @staticmethod
    def _validate_hash(
        value: str,
        *,
        name: str,
    ) -> None:

        if len(value) != 40 and name == "Base Commit SHA":
            raise ValueError(f"非法 {name}")

        if name == "Proposal ID" and len(value) != 64:
            raise ValueError(f"非法 {name}")

        if any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"非法 {name}")
