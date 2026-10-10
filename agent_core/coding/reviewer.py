import difflib
import hashlib
import json

from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
)
from agent_core.coding.review_model import ReviewPackage, ReviewFile


class PatchReviewError(ValueError):
    """Patch 审查准备失败。"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def calculate_proposal_id(
    proposal: FilePatchProposal,
) -> str:
    payload = {
        "schema_version": 1,
        "task_id": proposal.task_id,
        "artifact_id": proposal.artifact_id,
        "context_hash": proposal.context_hash,
        "patches": [
            {
                "relative_path": patch.relative_path,
                "expected_sha256": patch.expected_sha256,
                "after_sha256": patch.after_sha256,
                "before_content": patch.before_content,
                "after_content": patch.after_content,
                "explanation": patch.explanation,
            }
            for patch in sorted(
                proposal.patches,
                key=lambda item: item.relative_path,
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

        if proposal.status != "generated_not_applied":
            raise PatchReviewError("只能审查尚未应用的 Patch Proposal")

        if not proposal.patches:
            raise PatchReviewError("Patch Proposal 没有修改内容")

        seen_paths: set[str] = set()

        review_files: list[ReviewFile] = []

        for patch in proposal.patches:

            if patch.relative_path in seen_paths:
                raise PatchReviewError("存在重复修改文件: " f"{patch.relative_path}")

            seen_paths.add(patch.relative_path)

            actual_before_hash = sha256_text(patch.before_content)

            if actual_before_hash != patch.expected_sha256:
                raise PatchReviewError(
                    "修改前文件 Hash 不一致: " f"{patch.relative_path}"
                )

            actual_after_hash = sha256_text(patch.after_content)

            if actual_after_hash != patch.after_sha256:
                raise PatchReviewError(
                    "修改后文件 Hash 不一致: " f"{patch.relative_path}"
                )

            if patch.before_content == patch.after_content:
                raise PatchReviewError("没有实际代码修改: " f"{patch.relative_path}")

            expected_diff = "".join(
                difflib.unified_diff(
                    patch.before_content.splitlines(keepends=True),
                    patch.after_content.splitlines(keepends=True),
                    fromfile=("a/" + patch.relative_path),
                    tofile=("b/" + patch.relative_path),
                )
            )

            if expected_diff != patch.unified_diff:
                raise PatchReviewError(
                    "Diff 与实际修改内容不一致: " f"{patch.relative_path}"
                )

            review_files.append(
                ReviewFile(
                    relative_path=patch.relative_path,
                    expected_sha256=(patch.expected_sha256),
                    after_sha256=(patch.after_sha256),
                    explanation=patch.explanation,
                    unified_diff=expected_diff,
                )
            )

        review_files.sort(key=lambda item: item.relative_path)

        return ReviewPackage(
            proposal_id=calculate_proposal_id(proposal),
            task_id=proposal.task_id,
            artifact_id=proposal.artifact_id,
            context_hash=proposal.context_hash,
            files=review_files,
        )
