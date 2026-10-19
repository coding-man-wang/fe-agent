import re
from difflib import SequenceMatcher


class UnresolvedQuestionConsolidator:
    """
    把多个 Task 的 unresolved 问题收敛成 Case/Requirement 级问题。

    这里只做确定性的文本归一化与近似去重，不使用 Benchmark Oracle，
    也不重新做业务推理。
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.72,
        same_subject_threshold: float = 0.48,
        max_items: int = 8,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.same_subject_threshold = same_subject_threshold
        self.max_items = max_items

    def consolidate(self, items: list[str]) -> list[str]:
        candidates = [item.strip() for item in items if item and item.strip()]
        candidates.sort(key=self._information_score, reverse=True)

        chosen: list[str] = []
        for candidate in candidates:
            if any(self._same_question(candidate, current) for current in chosen):
                continue
            chosen.append(candidate)
            if len(chosen) >= self.max_items:
                break

        return sorted(chosen)

    def _same_question(self, left: str, right: str) -> bool:
        left_normalized = self._normalize(left)
        right_normalized = self._normalize(right)

        if left_normalized == right_normalized:
            return True

        ratio = SequenceMatcher(
            None,
            left_normalized,
            right_normalized,
        ).ratio()
        if ratio >= self.similarity_threshold:
            return True

        left_subjects = self._subjects(left)
        right_subjects = self._subjects(right)
        common_subjects = left_subjects & right_subjects

        return bool(common_subjects) and ratio >= self.same_subject_threshold

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = text.casefold().replace("\\", "/")
        normalized = re.sub(r"[，。；：、！？,.!?;:\s]+", "", normalized)
        normalized = re.sub(
            r"(尚未|仍未|当前未|未能|无法|需要|需|应|是否|具体|现有)",
            "",
            normalized,
        )
        return normalized

    @staticmethod
    def _subjects(text: str) -> set[str]:
        subjects = set(
            re.findall(
                r"[A-Za-z_][A-Za-z0-9_]*(?:\.(?:vue|ts|tsx|js|jsx))?",
                text,
            )
        )
        subjects.update(
            token
            for token in (
                "PaginationBar",
                "UsersView",
                "UserTable",
                "pageSize",
                "totalPages",
                "fetchUsers",
                "Playwright",
                "E2E",
                "Mock",
                "store",
                "防抖",
                "去重",
                "分页",
                "总页数",
            )
            if token.casefold() in text.casefold()
        )
        return {item.casefold() for item in subjects}

    @staticmethod
    def _information_score(text: str) -> tuple[int, int]:
        identifier_count = len(
            re.findall(r"[A-Za-z_][A-Za-z0-9_./-]*", text)
        )
        return identifier_count, len(text)
