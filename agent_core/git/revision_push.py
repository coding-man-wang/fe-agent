from pathlib import Path

from agent_core.git.approved_revision_executor import (
    ApprovedRevisionExecutionResult,
)
from agent_core.git.cli_client import (
    CliGitClient,
)
from agent_core.git.client import (
    GitClient,
)
from agent_core.git.pull_request_model import (
    PullRequestResult,
)
from agent_core.git.revision_push_model import (
    RevisionPushResult,
)
from agent_core.git.revision_task_model import (
    PreparedRevisionTask,
)


class RevisionPushError(RuntimeError):
    pass


class RevisionCASPushService:

    def __init__(
        self,
        *,
        repository_root: str | Path,
        git_client: GitClient | None = None,
    ) -> None:

        self.repository_root = Path(repository_root).resolve(strict=True)

        if not self.repository_root.is_dir():
            raise ValueError("repository_root 不是目录")

        self.git_client = git_client or CliGitClient()

    async def push(
        self,
        *,
        prepared: PreparedRevisionTask,
        execution: ApprovedRevisionExecutionResult,
        pull_request: PullRequestResult,
        remote_name: str = "origin",
    ) -> RevisionPushResult:

        self._validate_inputs(
            prepared=prepared,
            execution=execution,
            pull_request=pull_request,
            remote_name=remote_name,
        )

        expected_base = prepared.base_commit_sha

        new_commit = execution.commit_sha

        branch = pull_request.head_branch

        # ----------------------------------
        # 1. 新 Commit 必须真实存在于本地
        # ----------------------------------

        await self._require_commit_exists(new_commit)

        # ----------------------------------
        # 2. Remote 必须存在
        # ----------------------------------

        await self._require_remote_exists(remote_name)

        # ----------------------------------
        # 3. 查询远端 PR Branch 当前 HEAD
        # ----------------------------------

        remote_head = await self._read_remote_head(
            remote_name=remote_name,
            branch_name=branch,
        )

        if remote_head is None:

            raise RevisionPushError("Revision Push 要求原 PR " "远端分支已经存在")

        # ----------------------------------
        # 4. 幂等恢复
        #
        # 上一次可能已经 Push 成功，
        # 只是结果还没持久化。
        # ----------------------------------

        if remote_head == new_commit:

            return RevisionPushResult(
                provider=(pull_request.provider),
                repository=(pull_request.repository),
                pull_request_number=(pull_request.number),
                remote_name=remote_name,
                branch_name=branch,
                expected_base_commit_sha=(expected_base),
                pushed_commit_sha=(new_commit),
                status="already_applied",
            )

        # ----------------------------------
        # 5. CAS 前置检查
        # ----------------------------------

        if remote_head != expected_base:

            raise RevisionPushError(
                "远端 PR Branch 已变化："
                f"expected={expected_base}, "
                f"actual={remote_head}"
            )

        # ----------------------------------
        # 6. 精确 Push Commit SHA
        #
        # 注意：
        #
        # 不 Push private workspace branch 名。
        #
        # E
        # ↓
        # refs/heads/<original PR branch>
        # ----------------------------------

        result = await self.git_client.run(
            cwd=self.repository_root,
            args=[
                "push",
                remote_name,
                (f"{new_commit}:" f"refs/heads/{branch}"),
            ],
            allow_failure=True,
        )

        if result.exit_code != 0:

            raise RevisionPushError("Revision Push 失败: " f"{result.stderr.strip()}")

        # ----------------------------------
        # 7. Postcondition
        #
        # HTTP/Git 命令成功 ≠
        # 业务状态已经正确。
        # ----------------------------------

        actual_after_push = await self._read_remote_head(
            remote_name=remote_name,
            branch_name=branch,
        )

        if actual_after_push != new_commit:

            raise RevisionPushError(
                "Revision Push 后远端 HEAD "
                "与预期 Commit 不一致: "
                f"expected={new_commit}, "
                f"actual={actual_after_push}"
            )

        return RevisionPushResult(
            provider=(pull_request.provider),
            repository=(pull_request.repository),
            pull_request_number=(pull_request.number),
            remote_name=remote_name,
            branch_name=branch,
            expected_base_commit_sha=(expected_base),
            pushed_commit_sha=(new_commit),
            status="pushed",
        )

    def _validate_inputs(
        self,
        *,
        prepared: PreparedRevisionTask,
        execution: ApprovedRevisionExecutionResult,
        pull_request: PullRequestResult,
        remote_name: str,
    ) -> None:

        if not remote_name.strip():

            raise ValueError("remote_name 不能为空")

        if prepared.task_id != execution.task_id:

            raise RevisionPushError("Prepared Revision " "与 Execution Task 不一致")

        if prepared.base_commit_sha != execution.base_commit_sha:

            raise RevisionPushError(
                "Execution Base Commit " "与 Prepared Revision 不一致"
            )

        if pull_request.commit_sha != prepared.base_commit_sha:

            raise RevisionPushError(
                "本地 PullRequestResult " "已经不是 Revision 所基于的版本"
            )

        if not pull_request.head_branch:

            raise RevisionPushError("Pull Request " "缺少 Head Branch")

        if execution.commit_sha == prepared.base_commit_sha:

            raise RevisionPushError("Revision 没有产生新的 Commit")

        self._validate_sha(
            prepared.base_commit_sha,
            name="Base Commit SHA",
        )

        self._validate_sha(
            execution.commit_sha,
            name="Revision Commit SHA",
        )

    async def _require_commit_exists(
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

            raise RevisionPushError("Revision Commit " "在本地不存在: " f"{commit_sha}")

    async def _require_remote_exists(
        self,
        remote_name: str,
    ) -> None:

        result = await self.git_client.run(
            cwd=self.repository_root,
            args=[
                "remote",
                "get-url",
                remote_name,
            ],
            allow_failure=True,
        )

        if result.exit_code != 0:

            raise RevisionPushError("Git Remote 不存在: " f"{remote_name}")

    async def _read_remote_head(
        self,
        *,
        remote_name: str,
        branch_name: str,
    ) -> str | None:

        ref = f"refs/heads/" f"{branch_name}"

        result = await self.git_client.run(
            cwd=self.repository_root,
            args=[
                "ls-remote",
                "--heads",
                remote_name,
                ref,
            ],
            allow_failure=True,
        )

        if result.exit_code != 0:

            raise RevisionPushError("读取远端 Branch 失败: " f"{result.stderr.strip()}")

        output = result.stdout.strip()

        if not output:
            return None

        lines = [line.strip() for line in output.splitlines() if line.strip()]

        if len(lines) != 1:

            raise RevisionPushError("远端 Branch 查询结果异常")

        parts = lines[0].split()

        if len(parts) != 2:

            raise RevisionPushError("git ls-remote " "返回结构非法")

        sha, returned_ref = parts

        if returned_ref != ref:

            raise RevisionPushError("远端返回了非预期 Ref")

        self._validate_sha(
            sha,
            name="Remote Commit SHA",
        )

        return sha

    @staticmethod
    def _validate_sha(
        value: str,
        *,
        name: str,
    ) -> None:

        if len(value) != 40:

            raise RevisionPushError(f"非法 {name}")

        if any(char not in "0123456789abcdef" for char in value):

            raise RevisionPushError(f"非法 {name}")
