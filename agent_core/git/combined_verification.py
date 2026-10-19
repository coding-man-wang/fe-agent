from pathlib import Path

from agent_core.coding.verification_executor import (
    VerificationExecutor,
)
from agent_core.coding.verification_plan import (
    VerificationPlan,
)
from agent_core.git.cli_client import (
    CliGitClient,
)
from agent_core.git.client import (
    GitClient,
)
from agent_core.git.combined_verification_model import (
    CombinedVerificationPlan,
    CombinedVerificationResult,
)
from agent_core.git.parallel_integration_model import (
    ParallelIntegrationResult,
)


class CombinedVerificationError(ValueError):
    pass


class CombinedVerificationService:

    def __init__(
        self,
        *,
        executor: VerificationExecutor,
        git_client: GitClient | None = None,
    ) -> None:

        self.executor = executor

        self.git_client = git_client or CliGitClient()

    async def verify(
        self,
        *,
        integration: ParallelIntegrationResult,
        plan: CombinedVerificationPlan,
    ) -> CombinedVerificationResult:

        self._validate_inputs(
            integration=integration,
            plan=plan,
        )

        worktree = Path(integration.worktree_path).resolve(strict=True)

        if not worktree.is_dir():

            raise CombinedVerificationError("Integration Worktree " "不是目录")

        # ----------------------------------
        # 1. 验证实际 HEAD
        # ----------------------------------

        actual_head = await self._read_head(worktree)

        if actual_head != plan.commit_sha:

            raise CombinedVerificationError(
                "Integration Worktree HEAD " "与待验证 Commit 不一致"
            )

        # ----------------------------------
        # 2. 验证 Worktree clean
        # ----------------------------------

        await self._require_clean(worktree)

        # ----------------------------------
        # 3. 转成已有 Executor 能理解的 Plan
        # ----------------------------------

        executor_plan = VerificationPlan(
            proposal_id=(plan.integration_id),
            task_id=("integration:" f"{plan.integration_id}"),
            command_ids=(plan.command_ids),
        )

        result = await self.executor.run(plan=executor_plan)

        if result.proposal_id != plan.integration_id:

            raise CombinedVerificationError(
                "Verification Result " "Integration ID 不一致"
            )

        expected_task_id = "integration:" f"{plan.integration_id}"

        if result.task_id != expected_task_id:

            raise CombinedVerificationError(
                "Verification Result " "内部 Task ID 不一致"
            )

        # ----------------------------------
        # 4. Verification 执行结束后
        #    再检查一次 HEAD
        #
        # 防止验证脚本偷偷修改 Git 状态。
        # ----------------------------------

        actual_after = await self._read_head(worktree)

        if actual_after != plan.commit_sha:

            raise CombinedVerificationError(
                "Verification 执行期间 " "Integration HEAD 发生变化"
            )

        await self._require_clean(worktree)

        status = (
            "verified_integration"
            if result.status == "passed"
            else "integration_verification_failed"
        )

        return CombinedVerificationResult(
            integration_id=(plan.integration_id),
            commit_sha=(plan.commit_sha),
            status=status,
            checks=result.checks,
        )

    def _validate_inputs(
        self,
        *,
        integration: ParallelIntegrationResult,
        plan: CombinedVerificationPlan,
    ) -> None:

        if integration.status != "integrated_unverified":

            raise CombinedVerificationError(
                "只有 integrated_unverified " "状态允许 Combined Verification"
            )

        if integration.integration_id != plan.integration_id:

            raise CombinedVerificationError(
                "Verification Plan " "不属于当前 Integration"
            )

        if integration.final_commit_sha is None:

            raise CombinedVerificationError("Integration 缺少 " "Final Commit")

        if integration.final_commit_sha != plan.commit_sha:

            raise CombinedVerificationError(
                "Verification Plan " "Commit 与 Integration " "Final Commit 不一致"
            )

        self._validate_sha(plan.commit_sha)

    async def _read_head(
        self,
        worktree: Path,
    ) -> str:

        result = await self.git_client.run(
            cwd=worktree,
            args=[
                "rev-parse",
                "HEAD",
            ],
        )

        sha = result.stdout.strip()

        self._validate_sha(sha)

        return sha

    async def _require_clean(
        self,
        worktree: Path,
    ) -> None:

        result = await self.git_client.run(
            cwd=worktree,
            args=[
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
        )

        if result.stdout.strip():

            raise CombinedVerificationError("Integration Worktree " "不是 Clean 状态")

    @staticmethod
    def _validate_sha(
        value: str,
    ) -> None:

        if len(value) != 40:

            raise CombinedVerificationError("非法 Commit SHA")

        if any(char not in "0123456789abcdef" for char in value):

            raise CombinedVerificationError("非法 Commit SHA")
