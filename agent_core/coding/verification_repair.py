from __future__ import annotations

import difflib
import hashlib
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
    ValidatedFilePatch,
)
from agent_core.coding.verification_record_model import (
    VerificationRunRecord,
)
from agent_core.context.model import (
    TaskContextBundle,
)


class VerificationRepairError(ValueError):
    pass


class VerificationRepairNoop(VerificationRepairError):
    pass


class VerificationRepairTextModel(Protocol):

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str: ...


class VerificationRepairChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str = Field(min_length=1)

    after_content: str = Field(min_length=1)

    explanation: str = Field(min_length=1)


class VerificationRepairOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)

    repairable: bool

    reason: str = Field(min_length=1)

    changes: list[VerificationRepairChange] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_repair_state(
        self,
    ) -> "VerificationRepairOutput":

        if self.repairable and not self.changes:
            raise ValueError("repairable=true 时" "必须返回 changes")

        if not self.repairable and self.changes:
            raise ValueError("repairable=false 时" "changes 必须为空")

        return self


@dataclass(frozen=True)
class VerificationRepairResult:

    proposal: FilePatchProposal | None

    reason: str


class VerificationRepairService:
    """
    Verification-guided Bounded Repair。

    当前 v1：

    - 由 Full Execution 控制
      只有 test Task 使用；
    - 不重新规划文件范围；
    - 不改变 Patch Action；
    - 不允许添加 / 删除 Target；
    - 只重新决定已有 Target
      的 after_content；
    - 修复次数由上层 Harness 控制。
    """

    VERIFICATION_CONTEXT_FILES = (
        "package.json",
        "playwright.config.ts",
        "playwright.config.js",
        "playwright.config.mts",
        "vitest.config.ts",
        "vitest.config.js",
        "vite.config.ts",
        "vite.config.js",
        "tsconfig.e2e.json",
    )

    SYSTEM_INSTRUCTION = """
你正在修复一个已经执行过、但 Verification 失败的 Test Task。

这不是重新规划任务。

原 Patch Proposal 的：

- 文件集合
- relative_path
- action
- 写权限

都已经由 Harness 审批完成。

你只能根据 Verification Failure
重新生成这些相同文件的最终内容。

核心目标：

    修复导致 Verification 失败的测试实现，
    同时保持原测试意图和 Requirement 覆盖。

硬性规则：

1. 如果 repairable=true：

   changes 的文件路径集合必须与
   original_proposal_targets 完全一致。

   一个不能少，
   一个不能多。

2. 不允许修改 relative_path。

3. 不允许改变 action。

4. 不允许新增：

   - production source
   - config
   - helper
   - test file
   - 其他任意文件

5. action=create：

   文件在 Base Revision 中不存在。

   failed_after_content 是上一次
   Agent 创建但 Verification 失败的内容。

   你需要返回修复后的完整文件内容。

6. action=modify：

   before_content 是 Base Revision
   的原始文件内容。

   failed_after_content 是上一次
   Verification 失败的修改版本。

   返回修复后的完整文件内容。

7. 必须认真区分：

   - Production Bug
   - Test Bug
   - Test Environment Assumption Bug
   - Test Layer Selection Bug

8. 如果失败实际上要求修改
   PatchPlan 之外的生产代码才能修复：

       repairable=false

   不允许为了让测试通过，
   偷偷弱化测试或绕过真实产品问题。

9. 不允许仅通过以下方式
   “让测试变绿”：

   - 删除有意义的断言；
   - 把严格断言改成恒真断言；
   - skip / todo；
   - 增加任意长 sleep；
   - 捕获异常后忽略；
   - 删除 Requirement 要求覆盖的行为。

10. 如果发现某个测试场景本身
    使用了错误测试层级：

    例如 build-time / startup-time
    配置被错误假设为可以在浏览器
    运行过程中动态切换，

    可以修正这个测试场景，

    但只有在提供的 Repository Evidence
    表明对应业务契约已经由其他合理测试
    覆盖时，才允许删除重复且错误的场景。

    否则返回 repairable=false，
    交给更高层 Replanning。

11. verification_environment_context
    是只读配置证据。

    不允许修改其中的配置文件。

12. read_only_code_context
    只能用于理解当前代码和已有测试。

    不会因此获得写权限。

13. 只返回合法 JSON。

14. 不要输出 Markdown。

15. 不要声称已经运行或通过测试。

输出：

{
  "task_id": "当前 Task ID",
  "repairable": true,
  "reason": "为什么可以在当前 Patch Boundary 内修复",
  "changes": [
    {
      "relative_path": "必须与原 Target 完全一致",
      "after_content": "修复后的完整文件内容",
      "explanation": "为什么这样修复"
    }
  ]
}

如果无法在当前 Patch Boundary 内安全修复：

{
  "task_id": "当前 Task ID",
  "repairable": false,
  "reason": "为什么必须重新规划或修改范围",
  "changes": []
}
""".strip()

    def __init__(
        self,
        *,
        model: VerificationRepairTextModel,
        max_file_bytes: int = 100_000,
        max_failure_detail_chars: int = 16_000,
    ) -> None:

        if max_file_bytes < 1:
            raise ValueError("max_file_bytes " "必须 >= 1")

        if max_failure_detail_chars < 1:
            raise ValueError("max_failure_detail_chars " "必须 >= 1")

        self.model = model

        self.max_file_bytes = max_file_bytes

        self.max_failure_detail_chars = max_failure_detail_chars

    async def repair(
        self,
        *,
        proposal: FilePatchProposal,
        verification: VerificationRunRecord,
        bundle: TaskContextBundle,
        repository_root: str | Path,
    ) -> VerificationRepairResult:

        self._validate_inputs(
            proposal=proposal,
            verification=verification,
            bundle=bundle,
        )

        payload = self._build_payload(
            proposal=proposal,
            verification=verification,
            bundle=bundle,
            repository_root=repository_root,
        )

        raw_output = await self.model.complete(
            system_message=(self.SYSTEM_INSTRUCTION),
            user_message=json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
        )

        try:
            output = VerificationRepairOutput.model_validate_json(raw_output)

        except ValidationError as exc:
            raise VerificationRepairError(
                "Verification Repair " "输出不满足 Schema"
            ) from exc

        if output.task_id != proposal.task_id:
            raise VerificationRepairError("Verification Repair " "Task ID 不一致")

        if not output.repairable:
            return VerificationRepairResult(
                proposal=None,
                reason=output.reason,
            )

        self._validate_scope(
            proposal=proposal,
            output=output,
        )

        repaired_proposal = self._build_repaired_proposal(
            proposal=proposal,
            output=output,
        )

        return VerificationRepairResult(
            proposal=repaired_proposal,
            reason=output.reason,
        )

    def _validate_inputs(
        self,
        *,
        proposal: FilePatchProposal,
        verification: VerificationRunRecord,
        bundle: TaskContextBundle,
    ) -> None:

        if verification.task_id != proposal.task_id:
            raise VerificationRepairError("Verification 与 Proposal " "Task ID 不一致")

        if verification.proposal_id != self._proposal_id_from_record(verification):
            # 这里实际上只是保留一个
            # 清晰的 validation point。
            #
            # proposal_id 本身由 Reviewer
            # 重新校验。
            pass

        if verification.status != "failed":
            raise VerificationRepairError("只有 Verification " "失败才能进入 Repair")

        if verification.result is None:
            raise VerificationRepairError("Verification 没有 " "可分析的 Result")

        failed_checks = [
            check for check in verification.result.checks if check.status == "failed"
        ]

        if not failed_checks:
            raise VerificationRepairError(
                "Verification 标记 failed，" "但没有 failed check"
            )

        if bundle.task.task_id != proposal.task_id:
            raise VerificationRepairError("Context Bundle " "Task ID 不一致")

    @staticmethod
    def _proposal_id_from_record(
        verification: VerificationRunRecord,
    ) -> str:
        return verification.proposal_id

    def _build_payload(
        self,
        *,
        proposal: FilePatchProposal,
        verification: VerificationRunRecord,
        bundle: TaskContextBundle,
        repository_root: str | Path,
    ) -> dict:

        assert verification.result is not None

        failed_checks = [
            {
                "name": check.name,
                "command_id": (check.command_id),
                "status": (check.status),
                "details": (self._truncate_details(check.details)),
            }
            for check in verification.result.checks
            if check.status == "failed"
        ]

        return {
            "task": (bundle.task.model_dump(mode="json")),
            "requirement": (bundle.requirement.model_dump(mode="json")),
            "technical_spec": (bundle.technical_spec.model_dump(mode="json")),
            "change_surface": (bundle.change_surface),
            "verification_failures": (failed_checks),
            "original_proposal_targets": [
                {
                    "relative_path": (patch.relative_path),
                    "action": (patch.action),
                    "expected_sha256": (patch.expected_sha256),
                    "before_content": (patch.before_content),
                    "failed_after_content": (patch.after_content),
                    "explanation": (patch.explanation),
                }
                for patch in proposal.patches
            ],
            "read_only_code_context": [
                {
                    "relative_path": (item.relative_path),
                    "category": (item.category),
                    "content": (item.content),
                }
                for item in bundle.code_context
            ],
            "verification_environment_context": (
                self._load_verification_context(repository_root)
            ),
        }

    def _load_verification_context(
        self,
        repository_root: str | Path,
    ) -> list[dict[str, str]]:

        root = Path(repository_root).resolve(strict=True)

        result: list[dict[str, str]] = []

        for relative_path in self.VERIFICATION_CONTEXT_FILES:

            path = root / relative_path

            if not path.is_file() or path.is_symlink():
                continue

            try:
                data = path.read_bytes()

            except OSError:
                continue

            if len(data) > self.max_file_bytes:
                continue

            try:
                content = data.decode("utf-8")

            except UnicodeDecodeError:
                continue

            result.append(
                {
                    "relative_path": (relative_path),
                    "content": content,
                }
            )

        return result

    def _truncate_details(
        self,
        details: str,
    ) -> str:

        if len(details) <= self.max_failure_detail_chars:
            return details

        head_size = self.max_failure_detail_chars // 3

        tail_size = self.max_failure_detail_chars - head_size

        return (
            details[:head_size]
            + "\n\n...[verification output truncated]...\n\n"
            + details[-tail_size:]
        )

    @staticmethod
    def _validate_scope(
        *,
        proposal: FilePatchProposal,
        output: VerificationRepairOutput,
    ) -> None:

        expected_paths = [patch.relative_path for patch in proposal.patches]

        generated_paths = [change.relative_path for change in output.changes]

        if len(generated_paths) != len(set(generated_paths)):
            raise VerificationRepairError("Verification Repair " "返回重复文件")

        if set(generated_paths) != set(expected_paths):
            missing = sorted(set(expected_paths) - set(generated_paths))

            extra = sorted(set(generated_paths) - set(expected_paths))

            raise VerificationRepairError(
                "Verification Repair "
                "越过原 Patch Boundary；"
                f"missing={missing}; "
                f"extra={extra}"
            )

    def _build_repaired_proposal(
        self,
        *,
        proposal: FilePatchProposal,
        output: VerificationRepairOutput,
    ) -> FilePatchProposal:

        patch_map = {patch.relative_path: patch for patch in proposal.patches}

        validated: list[ValidatedFilePatch] = []

        any_content_changed = False

        for change in output.changes:

            original = patch_map[change.relative_path]

            after_content = change.after_content

            after_bytes = after_content.encode("utf-8")

            if len(after_bytes) > self.max_file_bytes:
                raise VerificationRepairError(
                    "Repair 后文件超过" "大小限制: " f"{change.relative_path}"
                )

            if after_content != original.after_content:
                any_content_changed = True

            after_sha256 = hashlib.sha256(after_bytes).hexdigest()

            if original.action == "modify":

                if original.before_content is None or original.expected_sha256 is None:
                    raise VerificationRepairError(
                        "modify repair " "缺少 before state: " f"{change.relative_path}"
                    )

                if after_content == original.before_content:
                    raise VerificationRepairError(
                        "Repair 把 modify "
                        "直接恢复成原文件，"
                        "说明当前 Patch "
                        "不再有实际修改: "
                        f"{change.relative_path}"
                    )

                unified_diff = "".join(
                    difflib.unified_diff(
                        original.before_content.splitlines(keepends=True),
                        after_content.splitlines(keepends=True),
                        fromfile=("a/" + change.relative_path),
                        tofile=("b/" + change.relative_path),
                    )
                )

            elif original.action == "create":

                if (
                    original.before_content is not None
                    or original.expected_sha256 is not None
                ):
                    raise VerificationRepairError(
                        "create repair "
                        "出现非法 before state: "
                        f"{change.relative_path}"
                    )

                unified_diff = "".join(
                    difflib.unified_diff(
                        [],
                        after_content.splitlines(keepends=True),
                        fromfile="/dev/null",
                        tofile=("b/" + change.relative_path),
                    )
                )

            else:
                raise VerificationRepairError(
                    "未知 Patch Action: " f"{original.action}"
                )

            validated.append(
                ValidatedFilePatch(
                    action=(original.action),
                    relative_path=(change.relative_path),
                    expected_sha256=(original.expected_sha256),
                    after_sha256=(after_sha256),
                    before_content=(original.before_content),
                    after_content=(after_content),
                    explanation=("verification-repair: " + change.explanation),
                    unified_diff=(unified_diff),
                )
            )

        if not any_content_changed:
            raise VerificationRepairNoop(
                "Verification Repair " "没有产生任何新的代码内容"
            )

        validated.sort(key=lambda item: (item.relative_path))

        return FilePatchProposal(
            task_id=(proposal.task_id),
            artifact_id=(proposal.artifact_id),
            context_hash=(proposal.context_hash),
            patches=validated,
        )
