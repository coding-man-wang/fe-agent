from pathlib import Path

from agent_core.git.errors import (
    GitWorkspaceConflictError,
)
from agent_core.git.repository import (
    GitRepositoryInspector,
)
from agent_core.git.workspace_binding_model import (
    WorkspaceExecutionContext,
)
from agent_core.git.workspace_model import (
    GitWorkspaceRecord,
)


class WorkspaceBindingService:
    """
    把一个 GitWorkspaceRecord
    转换成真正可用于执行的 Workspace Context。

    注意：
    GitWorkspaceRecord 是“记录”。

    WorkspaceExecutionContext 是：
    “刚刚重新验证过，可以放心执行”的结果。
    """

    def __init__(
            self,
            *,
            inspector: GitRepositoryInspector | None = None,
    ) -> None:

        self.inspector = inspector or GitRepositoryInspector()

    async def bind(
            self,
            *,
            workspace: GitWorkspaceRecord,
            task_id: str,
            proposal_id: str,
    ) -> WorkspaceExecutionContext:

        # 1. Workspace 必须已经创建成功。
        if workspace.status != "ready":
            raise GitWorkspaceConflictError("Workspace 当前不是 ready 状态")

        # 2. 防止把 TASK-A 的 Workspace
        #    拿去执行 TASK-B。
        if workspace.task_id != task_id:
            raise GitWorkspaceConflictError("Workspace Task ID 不一致")

        # 3. 防止批准的是 Proposal A，
        #    却拿 Proposal B 在这个 Workspace 执行。
        if workspace.proposal_id != proposal_id:
            raise GitWorkspaceConflictError("Workspace Proposal ID 不一致")

        execution_root = Path(workspace.worktree_path).resolve()

        source_root = Path(workspace.repository_root).resolve()

        if not execution_root.is_dir():
            raise GitWorkspaceConflictError("Workspace Worktree 不存在")

        # 4. 再问 Git 一遍：
        #    “这个目录现在到底是什么状态？”
        state = await self.inspector.inspect(execution_root)

        actual_root = Path(state.repository_root).resolve()

        if actual_root != execution_root:
            raise GitWorkspaceConflictError("Workspace Git Root 不一致")

        if state.branch_name != workspace.branch_name:
            raise GitWorkspaceConflictError("Workspace Branch 已发生变化")

        return WorkspaceExecutionContext(
            workspace_id=(workspace.workspace_id),
            task_id=workspace.task_id,
            proposal_id=(workspace.proposal_id),
            source_repository_root=str(source_root),
            execution_root=str(execution_root),
            base_commit_sha=(workspace.base_commit_sha),
            branch_name=(workspace.branch_name),
        )
