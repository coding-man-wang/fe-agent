import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
)

RetrievalQuerySource = Literal[
    "raw_requirement",
    "requirement_spec",
    "task",
]


RetrievalQueryKind = Literal[
    "literal",
    "token",
    "text",
    "module",
    "input",
]

NEGATIVE_CLAUSE_PATTERNS = (
    re.compile(
        r"不修改",
        re.IGNORECASE,
    ),
    re.compile(
        r"不要修改",
        re.IGNORECASE,
    ),
    re.compile(
        r"不得",
        re.IGNORECASE,
    ),
    re.compile(
        r"无需",
        re.IGNORECASE,
    ),
    re.compile(
        r"不需要",
        re.IGNORECASE,
    ),
    re.compile(
        r"保持.+不变",
        re.IGNORECASE,
    ),
)


CLAUSE_SPLITTER = re.compile(r"[。；;\n]+")


class RetrievalQuery(BaseModel):
    """
    一条真正用于代码搜索的 Query。

    text:
        实际搜索内容。

    sources:
        这个搜索词是从哪里来的。

    kinds:
        它属于哪种信息。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    text: str = Field(
        min_length=1,
    )

    sources: list[RetrievalQuerySource] = Field(
        min_length=1,
    )

    kinds: list[RetrievalQueryKind] = Field(
        min_length=1,
    )


class RetrievalQueryBuilder:
    """
    Retrieval Query Builder v0.2。

    核心目标：

    不再只依赖 Task。

    同时保留：

    1. 用户原始 Requirement
    2. RequirementSpec
    3. TaskDefinition

    并记录每个 Query 的来源。
    """

    def __init__(
        self,
        *,
        max_queries: int = 6,
    ) -> None:

        if max_queries < 1:
            raise ValueError("max_queries 必须 >= 1")

        self.max_queries = max_queries

    def build(
        self,
        *,
        task: TaskDefinition,
        raw_requirement: str | None = None,
        requirement_spec: RequirementSpec | None = None,
    ) -> list[RetrievalQuery]:

        raw_queries = self._build_raw_requirement_queries(raw_requirement)

        requirement_queries = self._build_requirement_spec_queries(requirement_spec)

        task_queries = self._build_task_queries(task)

        # -----------------------------------
        # 不让某一路信息霸占所有 Query。
        #
        # 采用轮询方式：
        #
        # raw
        # requirement
        # task
        # raw
        # requirement
        # task
        # -----------------------------------

        groups = [
            raw_queries,
            requirement_queries,
            task_queries,
        ]

        interleaved: list[RetrievalQuery] = []

        max_group_length = max(
            (len(group) for group in groups),
            default=0,
        )

        for index in range(max_group_length):

            for group in groups:

                if index >= len(group):
                    continue

                interleaved.append(group[index])

        # -----------------------------------
        # 去重。
        #
        # 如果同一个词同时来源于：
        #
        # RequirementSpec
        # Task
        #
        # 不直接丢掉其中一个来源，
        # 而是把 sources 合并。
        # -----------------------------------

        merged: dict[
            str,
            RetrievalQuery,
        ] = {}

        order: list[str] = []

        for query in interleaved:

            normalized = query.text.strip().lower()

            if not normalized:
                continue

            existing = merged.get(normalized)

            if existing is None:

                merged[normalized] = query.model_copy(deep=True)

                order.append(normalized)

                continue

            sources = list(existing.sources)

            for source in query.sources:

                if source not in sources:
                    sources.append(source)

            kinds = list(existing.kinds)

            for kind in query.kinds:

                if kind not in kinds:
                    kinds.append(kind)

            merged[normalized] = existing.model_copy(
                deep=True,
                update={
                    "sources": sources,
                    "kinds": kinds,
                },
            )

        result = [merged[key] for key in order]

        result = result[: self.max_queries]

        if not result:

            raise ValueError("无法生成 Retrieval Query")

        return result

    @staticmethod
    def _extract_positive_segments(
        text: str,
    ) -> list[str]:
        """
        从需求文本里提取适合做
        正向代码搜索的部分。

        例如：

        调整用户名最大长度为 30。
        不修改 API、Router、Store。

        最终只保留：

        调整用户名最大长度为 30
        """

        segments = [
            segment.strip()
            for segment in CLAUSE_SPLITTER.split(text)
            if segment.strip()
        ]

        positive: list[str] = []

        for segment in segments:

            is_negative = any(
                pattern.search(segment) for pattern in NEGATIVE_CLAUSE_PATTERNS
            )

            if is_negative:
                continue

            positive.append(segment)

        return positive

    def _build_raw_requirement_queries(
        self,
        raw_requirement: str | None,
    ) -> list[RetrievalQuery]:

        if not raw_requirement:
            return []

        requirement = raw_requirement.strip()

        if not requirement:
            return []

        result: list[RetrievalQuery] = []

        # -----------------------------------
        # 1. 优先提取用户明确写出的字符串。
        #
        # 例如：
        #
        # "Search users"
        # "Page 2 / 5"
        #
        # 这种信息对于代码搜索非常有价值。
        # -----------------------------------

        literals = self._extract_literals(requirement)

        for literal in literals:

            result.append(
                RetrievalQuery(
                    text=literal,
                    sources=["raw_requirement"],
                    kinds=["literal"],
                )
            )

            # --------------------------------
            # 对 Literal 再拆一些英文关键词。
            #
            # "Page 2 / 5"
            #
            # 可以进一步得到：
            #
            # Page
            #
            # Exact Search 更容易命中源码。
            # --------------------------------

            tokens = self._extract_literal_tokens(literal)

            for token in tokens:

                result.append(
                    RetrievalQuery(
                        text=token,
                        sources=["raw_requirement"],
                        kinds=["token"],
                    )
                )

        # -----------------------------------
        # 2. 保留完整原始 Requirement。
        #
        # 对当前 Exact Search 帮助可能有限，
        # 但下一阶段 Semantic Search
        # 会非常有价值。
        # -----------------------------------

        positive_segments = self._extract_positive_segments(requirement)

        for segment in positive_segments:
            result.append(
                RetrievalQuery(
                    text=segment,
                    sources=["raw_requirement"],
                    kinds=["text"],
                )
            )

        return result

    def _build_requirement_spec_queries(
        self,
        requirement_spec: RequirementSpec | None,
    ) -> list[RetrievalQuery]:

        if requirement_spec is None:
            return []

        candidates = [
            requirement_spec.title,
            requirement_spec.summary,
            *requirement_spec.goals,
            *requirement_spec.acceptance_criteria,
        ]

        result: list[RetrievalQuery] = []

        for candidate in candidates:

            text = candidate.strip()

            if not text:
                continue

            segments = self._extract_positive_segments(text)

            for segment in segments:
                result.append(
                    RetrievalQuery(
                        text=segment,
                        sources=["requirement_spec"],
                        kinds=["text"],
                    )
                )

        # -----------------------------------
        # 注意：
        #
        # 这里暂时没有使用 constraints
        # 和 non_goals。
        #
        # 原因：
        #
        # "不要修改 API"
        #
        # 不应该被转成：
        #
        # search("API")
        #
        # 否则反而会把无关 API 文件
        # 搜进 Context。
        # -----------------------------------

        return result

    def _build_task_queries(
        self,
        task: TaskDefinition,
    ) -> list[RetrievalQuery]:

        result: list[RetrievalQuery] = []

        # Task 自身表达
        # 先于 target_modules。
        #
        # 避免过度依赖 Planner
        # 猜出来的模块路径。

        for candidate in [
            task.title,
            task.objective,
        ]:

            text = candidate.strip()

            if not text:
                continue

            result.append(
                RetrievalQuery(
                    text=text,
                    sources=["task"],
                    kinds=["text"],
                )
            )

        for module in task.target_modules:

            text = module.strip()

            if not text:
                continue

            result.append(
                RetrievalQuery(
                    text=text,
                    sources=["task"],
                    kinds=["module"],
                )
            )

        for task_input in task.inputs:

            text = task_input.strip()

            if not text:
                continue

            result.append(
                RetrievalQuery(
                    text=text,
                    sources=["task"],
                    kinds=["input"],
                )
            )

        return result

    @staticmethod
    def _extract_literals(
        text: str,
    ) -> list[str]:

        patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r"“([^”]+)”",
            r"‘([^’]+)’",
        ]

        result: list[str] = []

        seen: set[str] = set()

        for pattern in patterns:

            for match in re.findall(
                pattern,
                text,
            ):

                value = match.strip()

                if not value:
                    continue

                normalized = value.lower()

                if normalized in seen:
                    continue

                seen.add(normalized)

                result.append(value)

        return result

    @staticmethod
    def _extract_literal_tokens(
        literal: str,
    ) -> list[str]:

        candidates = re.findall(
            (r"[A-Za-z]" r"[A-Za-z0-9_.:/-]*"),
            literal,
        )

        result: list[str] = []

        seen: set[str] = set()

        for token in candidates:

            # 太短的英文词搜索价值低，
            # 反而容易产生大量噪声。
            if len(token) < 4:
                continue

            normalized = token.lower()

            if normalized in seen:
                continue

            seen.add(normalized)

            result.append(token)

        return result
