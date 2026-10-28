import difflib
import hashlib
import json

from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
    ValidatedFilePatch,
)
from agent_core.coding.review_model import (
    ReviewFile,
    ReviewPackage,
)


class PatchReviewError(ValueError):
    """Patch 审查准备失败。"""


def sha256_text(
    text: str,
) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def calculate_proposal_id(
    proposal: FilePatchProposal,
) -> str:
    """
    根据 Proposal 的完整内容计算稳定 ID。

    注意：
    action 必须进入 Proposal Identity。

    因为：

        modify foo.ts
        create foo.ts

    虽然 path 相同，
    但属于完全不同的 Side Effect。
    """

    payload = {
        "schema_version": 2,
        "task_id": (proposal.task_id),
        "artifact_id": (proposal.artifact_id),
        "context_hash": (proposal.context_hash),
        "patches": [
            {
                "relative_path": (patch.relative_path),
                "action": (patch.action),
                "expected_sha256": (patch.expected_sha256),
                "after_sha256": (patch.after_sha256),
                "before_content": (patch.before_content),
                "after_content": (patch.after_content),
                "explanation": (patch.explanation),
            }
            for patch in sorted(
                proposal.patches,
                key=lambda item: (item.relative_path),
            )
        ],
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class PatchReviewer:

    def prepare(
        self,
        proposal: FilePatchProposal,
    ) -> ReviewPackage:
        """
        把已经生成但尚未 Apply 的 Proposal
        转换为可审批的 ReviewPackage。

        当前支持：

            modify existing file
            create new file

        Review 阶段不产生任何真实文件副作用。
        """

        if proposal.status != "generated_not_applied":
            raise PatchReviewError("只能审查尚未应用的 " "Patch Proposal")

        if not proposal.patches:
            raise PatchReviewError("Patch Proposal " "没有修改内容")

        seen_paths: set[str] = set()

        review_files: list[ReviewFile] = []

        for patch in proposal.patches:

            if patch.relative_path in seen_paths:
                raise PatchReviewError("存在重复修改文件: " f"{patch.relative_path}")

            seen_paths.add(patch.relative_path)

            review_files.append(self._review_patch(patch))

        review_files.sort(key=lambda item: (item.relative_path))

        return ReviewPackage(
            proposal_id=(calculate_proposal_id(proposal)),
            task_id=(proposal.task_id),
            artifact_id=(proposal.artifact_id),
            context_hash=(proposal.context_hash),
            files=review_files,
        )

    def _review_patch(
        self,
        patch: ValidatedFilePatch,
    ) -> ReviewFile:
        """
        对一个已经经过 PatchGenerator
        验证的 File Patch 再做独立 Review 校验。

        Reviewer 不信任 Generator 的结果，
        会重新：

        1. 校验 Action 对应的 Before State；
        2. 重新计算 Hash；
        3. 重新生成 Diff；
        4. 与 Proposal 中的 Diff 对账。
        """

        # ==================================
        # 1. After State
        #
        # modify / create 都必须拥有
        # 一个真实的 after_content。
        # ==================================

        actual_after_hash = sha256_text(patch.after_content)

        if actual_after_hash != patch.after_sha256:
            raise PatchReviewError("修改后文件 Hash 不一致: " f"{patch.relative_path}")

        # ==================================
        # 2. Action-specific Review
        # ==================================

        if patch.action == "modify":

            expected_diff = self._review_modify_patch(patch)

        elif patch.action == "create":

            expected_diff = self._review_create_patch(patch)

        else:
            raise PatchReviewError(
                "未知 Patch Action: "
                f"{patch.action}; "
                f"path="
                f"{patch.relative_path}"
            )

        # ==================================
        # 3. Diff Integrity
        #
        # Generator 给出的 diff
        # 必须和 Reviewer 独立计算出的
        # diff 完全一致。
        # ==================================

        if expected_diff != patch.unified_diff:
            raise PatchReviewError(
                "Diff 与实际修改内容" "不一致: " f"{patch.relative_path}"
            )

        # ==================================
        # 4. Review Package File
        # ==================================

        return ReviewFile(
            relative_path=(patch.relative_path),
            action=(patch.action),
            expected_sha256=(patch.expected_sha256),
            after_sha256=(patch.after_sha256),
            explanation=(patch.explanation),
            unified_diff=(expected_diff),
        )

    @staticmethod
    def _review_modify_patch(
        patch: ValidatedFilePatch,
    ) -> str:
        """
        modify：

            existing before
                ↓
            modified after

        必须存在：

            before_content
            expected_sha256
        """

        if patch.before_content is None:
            raise PatchReviewError(
                "modify patch " "缺少 before_content: " f"{patch.relative_path}"
            )

        if patch.expected_sha256 is None:
            raise PatchReviewError(
                "modify patch " "缺少 expected_sha256: " f"{patch.relative_path}"
            )

        actual_before_hash = sha256_text(patch.before_content)

        if actual_before_hash != patch.expected_sha256:
            raise PatchReviewError(
                "修改前文件 Hash " "不一致: " f"{patch.relative_path}"
            )

        if patch.before_content == patch.after_content:
            raise PatchReviewError("没有实际代码修改: " f"{patch.relative_path}")

        return "".join(
            difflib.unified_diff(
                patch.before_content.splitlines(keepends=True),
                patch.after_content.splitlines(keepends=True),
                fromfile=("a/" + patch.relative_path),
                tofile=("b/" + patch.relative_path),
            )
        )

    @staticmethod
    def _review_create_patch(
        patch: ValidatedFilePatch,
    ) -> str:
        """
        create：

            absent
                ↓
            new file

        因此：

            expected_sha256 = None
            before_content = None

        Diff：

            /dev/null
                ↓
            b/new-file
        """

        if patch.expected_sha256 is not None:
            raise PatchReviewError(
                "create patch " "不能存在 " "expected_sha256: " f"{patch.relative_path}"
            )

        if patch.before_content is not None:
            raise PatchReviewError(
                "create patch " "不能存在 " "before_content: " f"{patch.relative_path}"
            )

        if not patch.after_content:
            raise PatchReviewError(
                "create patch " "不能创建空内容文件: " f"{patch.relative_path}"
            )

        return "".join(
            difflib.unified_diff(
                [],
                patch.after_content.splitlines(keepends=True),
                fromfile="/dev/null",
                tofile=("b/" + patch.relative_path),
            )
        )
