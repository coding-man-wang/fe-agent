from __future__ import annotations

import difflib
import hashlib
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Literal

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
from agent_core.policies.model import (
    EffectivePolicy,
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


class VerificationRepairCorrectionFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)

    relative_path: str = Field(min_length=1)

    message: str = Field(min_length=1)


VerificationRepairMode = Literal[
    "structural_only",
    "behavioral",
    "mixed",
]


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

    STRUCTURAL_COMMAND_IDS = {
        "frontend-lint",
        "frontend-typecheck",
        "frontend-build",
    }

    BEHAVIORAL_COMMAND_IDS = {
        "frontend-unit-test",
        "frontend-e2e",
    }

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

16. effective_policy 是当前 Task
    已由外部 Policy Resolver
    计算完成的工程规则。

17. severity=must 的规则
    在 Repair 后仍必须成立。

18. Repair 不允许为了修复 Verification
    而违反原 Effective Policy。

19. Effective Policy
    不能扩大原 Patch Boundary，
    不能新增 Target，
    不能改变 action。
20. repair_correction_feedback
    表示上一份 Repair Proposal
    已经被确定性的 Harness Gate 拒绝。

21. correction feedback 不是建议，
    是本轮重新生成必须满足的约束。

22. 如果 feedback 包含：

        repair_test_case_removed

    则必须保留对应 Test Case，
    包括原有 Test Title。

    应修正 Test Implementation
    或 Proof Strategy，
    而不是删除、跳过、重命名场景。

23. 如果 feedback 包含：

        repair_assertion_reduced

    则不得通过减少 assertion
    数量解决问题。

    可以重写错误 assertion，
    但必须继续证明相同或更强的
    Behavioral Obligation。

24. 如果某个 assertion 的证明方式
    与当前 Runtime Path 不匹配：

    例如 Mock 模式下错误地通过
    HTTP request count
    证明数据 reload，

    可以改变 Proof Strategy，

    但不得删除该 Requirement
    所对应的验证义务。

25. 如果无法在：
        原 Patch Boundary
        +
        原 Test Case
        +
        原 Verification Obligation
    下安全修复：

        repairable=false

    不允许继续弱化测试。
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

    STRUCTURAL_ONLY_INSTRUCTION = """
    当前 Repair Mode：

        structural_only

    说明当前 Verification Failure
    只来自结构性检查，例如：

    - lint
    - typecheck
    - build

    因此本次 Repair
    只能修复导致这些结构性错误的问题。

    硬性约束：

    1. 不得改变已有 Test Case
       所验证的业务行为。

    2. 不得删除、合并或重命名
       已有 Test Case。

    3. 不得减少 assertion。

    4. 不得为了消除 lint/typecheck
       而修改 assertion 的业务预期。

    5. 不得修改：

       - expected business value
       - filter semantics
       - mock business result
       - user-visible behavior expectation

    6. 应优先只修复：

       - TypeScript type annotation
       - import
       - mock typing
       - spy typing
       - helper typing
       - unsafe cast
       - module typing
       - syntax / lint violation

    7. 对 no-explicit-any：

       优先使用：

       - 已有 Repository Type；
       - 类型推导；
       - unknown；
       - Parameters<T>；
       - ReturnType<T>；
       - 明确 interface / type；
       - 测试框架已有的 typed mock API。

       不要通过：

       - eslint-disable
       - @ts-ignore
       - @ts-nocheck

       绕过工程规范。

    8. 如果无法在完全保留
       Test Behavioral Contract 的情况下
       修复结构问题：

           repairable=false

       不允许通过改变测试语义获得 green。

    本次目标不是重新设计测试。

    本次目标只是：

        修复 Structural Verification Failure。
    """.strip()

    BEHAVIORAL_INSTRUCTION = """
    当前 Repair Mode：

        behavioral

    Verification Failure
    来自 Unit Test 或 E2E Test。

    可以修复：

    - Mock 初始化顺序
    - Module isolation
    - Import timing
    - Async waiting
    - Locator
    - Fixture
    - Test environment setup
    - 合法的 behavioral assertion implementation

    但仍必须遵守：

    - 原 Patch Boundary；
    - Effective Policy；
    - Verification Integrity Guard；
    - 不允许通过 skip/todo
      或降低 Verification Obligation
      获得 green。
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
        effective_policy: EffectivePolicy | None = None,
        correction_feedback: list[VerificationRepairCorrectionFeedback] | None = None,
    ) -> VerificationRepairResult:

        self._validate_inputs(
            proposal=proposal,
            verification=verification,
            bundle=bundle,
        )

        repair_mode = self._classify_repair_mode(verification)

        payload = self._build_payload(
            proposal=proposal,
            verification=verification,
            bundle=bundle,
            repository_root=(repository_root),
            effective_policy=(effective_policy),
            repair_mode=repair_mode,
            correction_feedback=(correction_feedback or []),
        )

        system_message = self.SYSTEM_INSTRUCTION

        if repair_mode == "structural_only":
            system_message = system_message + "\n\n" + self.STRUCTURAL_ONLY_INSTRUCTION

        elif repair_mode == "behavioral":
            system_message = system_message + "\n\n" + self.BEHAVIORAL_INSTRUCTION

        raw_output = await self.model.complete(
            system_message=(system_message),
            user_message=(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                )
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

    @classmethod
    def _classify_repair_mode(
        cls,
        verification: VerificationRunRecord,
    ) -> VerificationRepairMode:

        if verification.result is None:
            raise VerificationRepairError("Verification 缺少 Result")

        failed_command_ids = {
            check.command_id
            for check in verification.result.checks
            if (check.status == "failed" and check.command_id is not None)
        }

        if not failed_command_ids:
            raise VerificationRepairError("Verification Failure " "缺少 command_id")

        if failed_command_ids <= cls.STRUCTURAL_COMMAND_IDS:
            return "structural_only"

        if failed_command_ids <= cls.BEHAVIORAL_COMMAND_IDS:
            return "behavioral"

        return "mixed"

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
        effective_policy: EffectivePolicy | None,
        repair_mode: VerificationRepairMode,
        correction_feedback: list[VerificationRepairCorrectionFeedback],
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
            "effective_policy": (
                effective_policy.model_dump(mode="json")
                if (effective_policy is not None)
                else {"rules": []}
            ),
            "repair_mode": (repair_mode),
            "repair_correction_feedback": [
                item.model_dump(mode="json") for item in correction_feedback
            ],
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
