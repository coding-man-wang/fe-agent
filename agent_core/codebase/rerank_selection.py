from agent_core.codebase.reranker import (
    RerankHit,
)


class AdaptiveRerankSelectionPolicy:
    """
    根据 Reranker 排名结果，
    自动判断应该保留多少文件。

    第一版原则：

    1. 至少保留 min_files
    2. 最多保留 max_files
    3. 如果相邻结果出现明显 score gap，
       在 gap 前截断
    """

    def __init__(
        self,
        *,
        min_files: int = 2,
        max_files: int = 8,
        min_score_gap: float = 0.20,
    ) -> None:

        if min_files < 1:
            raise ValueError("min_files 必须 >= 1")

        if max_files < min_files:
            raise ValueError("max_files 必须 >= min_files")

        if min_score_gap < 0:
            raise ValueError("min_score_gap 不能小于 0")

        self.min_files = min_files
        self.max_files = max_files
        self.min_score_gap = min_score_gap

    def select(
        self,
        hits: list[RerankHit],
    ) -> list[RerankHit]:

        if not hits:
            return []

        limited = hits[: self.max_files]

        if len(limited) <= self.min_files:
            return limited

        # 从至少保留 min_files 的位置
        # 开始寻找明显的分数断层。
        for index in range(
            self.min_files - 1,
            len(limited) - 1,
        ):

            current = limited[index]

            next_item = limited[index + 1]

            gap = current.score - next_item.score

            if gap >= self.min_score_gap:

                return limited[: index + 1]

        # 没有明显断层：
        # 不冒险删文件。
        return limited
