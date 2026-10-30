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

RetrievalIntent = Literal[
    # 我要找到这次真正可能需要处理的实现位置
    "implementation_location",
    # 我要理解调用链、数据流、依赖关系
    "dependency_context",
    # 我要找仓库里已有的类似实现作为参考
    "similar_implementation",
    # 我要找测试模式 / 已有测试
    "test_pattern",
    # 我要找工程规范、构建、验证规则
    "project_policy",
]

SIMILAR_IMPLEMENTATION_PATTERN = re.compile(
    (r"参考|复用|类似|相似|沿用|" r"现有.+实现|" r"reuse|similar|reference"),
    re.IGNORECASE,
)


TEST_INTENT_PATTERN = re.compile(
    (r"测试|单测|单元测试|端到端|" r"e2e|playwright|vitest|" r"unit test"),
    re.IGNORECASE,
)


PROJECT_POLICY_PATTERN = re.compile(
    (r"lint|typecheck|build|ci|" r"工程门禁|质量门禁|" r"工程规范"),
    re.IGNORECASE,
)

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
        Query 来源于哪一层。

    kinds:
        Query 的信息形态。

    intents:
        为什么要执行这次搜索。
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

    intents: list[RetrievalIntent] = Field(
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

            intents = list(existing.intents)

            for intent in query.intents:

                if intent not in intents:
                    intents.append(intent)

            merged[normalized] = existing.model_copy(
                deep=True,
                update={
                    "sources": sources,
                    "kinds": kinds,
                    "intents": intents,
                },
            )

        result = [merged[key] for key in order]

        result = self._select_queries_by_intent(result)

        if not result:

            raise ValueError("无法生成 Retrieval Query")

        return result

    @staticmethod
    def _extract_positive_segments(
        text: str,
    ) -> list[str]:
        """
        从文本中提取适合做正向 Retrieval
        的内容。

        关键规则：

        1. 纯正向句：
           直接保留

        2. 纯负向句：
           丢弃

        3. 正向 + 负向混在一句：
           只保留负向约束之前的正向部分

        例如：

        "请优先复用现有 pageSize 实现，
         不修改 API 请求协议"

        最终保留：

        "请优先复用现有 pageSize 实现"
        """

        segments = [
            segment.strip()
            for segment in CLAUSE_SPLITTER.split(text)
            if segment.strip()
        ]

        positive: list[str] = []

        for segment in segments:

            negative_starts: list[int] = []

            for pattern in NEGATIVE_CLAUSE_PATTERNS:

                match = pattern.search(segment)

                if match is not None:
                    negative_starts.append(match.start())

            # 完全没有负向约束
            if not negative_starts:
                positive.append(segment)

                continue

            # 找最早出现的负向约束
            first_negative = min(negative_starts)

            # 负向约束就在句首：
            # 整句都不适合拿去正向搜索
            if first_negative == 0:
                continue

            # 保留负向约束之前的内容
            prefix = segment[:first_negative].rstrip("，, ").strip()

            if prefix:
                positive.append(prefix)

        return positive

    def _select_queries_by_intent(
        self,
        queries: list[RetrievalQuery],
    ) -> list[RetrievalQuery]:

        if len(queries) <= self.max_queries:
            return queries

        selected: list[RetrievalQuery] = []

        selected_indexes: set[int] = set()

        covered_intents: set[RetrievalIntent] = set()

        # ----------------------------------
        # 第一轮：
        #
        # 优先挑选能带来新 Intent 的 Query。
        #
        # 比如：
        #
        # implementation
        # dependency
        # similar
        # test
        #
        # 不让 6 个 Query 全部只是
        # implementation_location。
        # ----------------------------------

        for index, query in enumerate(queries):

            has_new_intent = any(
                intent not in covered_intents for intent in query.intents
            )

            if not has_new_intent:
                continue

            selected.append(query)

            selected_indexes.add(index)

            covered_intents.update(query.intents)

            if len(selected) >= self.max_queries:
                return selected

        # ----------------------------------
        # 第二轮：
        #
        # Intent 覆盖完成后，
        # 再按照原来的稳定顺序补满。
        # ----------------------------------

        for index, query in enumerate(queries):

            if index in selected_indexes:
                continue

            selected.append(query)

            if len(selected) >= self.max_queries:
                break

        return selected

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
                    intents=[
                        "implementation_location",
                    ],
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
                        intents=[
                            "implementation_location",
                        ],
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
                    intents=self._infer_text_intents(
                        text=segment,
                        default_intent=("implementation_location"),
                    ),
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
                        intents=self._infer_text_intents(
                            text=segment,
                            default_intent=("implementation_location"),
                        ),
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

    @staticmethod
    def _default_task_intent(
        task: TaskDefinition,
    ) -> RetrievalIntent:

        mapping: dict[
            str,
            RetrievalIntent,
        ] = {
            "code_change": ("implementation_location"),
            "investigation": ("dependency_context"),
            "test": ("test_pattern"),
            "verification": ("project_policy"),
        }

        return mapping[task.task_type]

    def _build_task_queries(
        self,
        task: TaskDefinition,
    ) -> list[RetrievalQuery]:

        result: list[RetrievalQuery] = []

        default_intent = self._default_task_intent(task)

        # ==================================
        # 1. Logical Responsibilities
        #
        # v1.3 的主要 Task Retrieval Signal。
        #
        # 它描述：
        #   “需要解决什么责任”
        #
        # 而不是：
        #   “猜哪个文件”
        # ==================================

        for responsibility in task.responsibilities:

            text = responsibility.strip()

            if not text:
                continue

            result.append(
                RetrievalQuery(
                    text=text,
                    sources=["task"],
                    kinds=["text"],
                    intents=(
                        self._infer_text_intents(
                            text=text,
                            default_intent=(default_intent),
                        )
                    ),
                )
            )

        # ==================================
        # 2. Investigation Agenda
        #
        # 调查问题本身就是很好的
        # Retrieval Query。
        # ==================================

        for question in task.investigation_questions:

            text = question.strip()

            if not text:
                continue

            intents: list[RetrievalIntent] = [
                "dependency_context",
            ]

            if task.task_type in {
                "investigation",
                "code_change",
            }:
                intents.append("implementation_location")

            result.append(
                RetrievalQuery(
                    text=text,
                    sources=["task"],
                    kinds=["text"],
                    intents=intents,
                )
            )

        # ==================================
        # 3. Task Summary Fallback
        #
        # 保留 title / objective，
        # 避免旧 Plan 或非常短的 Logical
        # Plan 丢失检索能力。
        # ==================================

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
                    intents=(
                        self._infer_text_intents(
                            text=text,
                            default_intent=(default_intent),
                        )
                    ),
                )
            )

        # ==================================
        # 4. Repository Hint
        #
        # 新 Plan:
        #   repository_hints
        #
        # 旧 Frozen Plan:
        #   target_modules
        #
        # 统一从 Compatibility API 获取。
        # ==================================

        for hint in task.repository_search_hints():

            text = hint.strip()

            if not text:
                continue

            result.append(
                RetrievalQuery(
                    text=text,
                    sources=["task"],
                    kinds=["module"],
                    intents=[
                        "implementation_location",
                    ],
                )
            )

        # ==================================
        # 5. Workflow Inputs
        # ==================================

        for task_input in task.inputs:

            text = task_input.strip()

            if not text:
                continue

            result.append(
                RetrievalQuery(
                    text=text,
                    sources=["task"],
                    kinds=["input"],
                    intents=[
                        "dependency_context",
                    ],
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

    @staticmethod
    def _append_intent(
        intents: list[RetrievalIntent],
        intent: RetrievalIntent,
    ) -> None:

        if intent not in intents:
            intents.append(intent)

    @classmethod
    def _infer_text_intents(
        cls,
        *,
        text: str,
        default_intent: RetrievalIntent,
    ) -> list[RetrievalIntent]:

        intents: list[RetrievalIntent] = [
            default_intent,
        ]

        if SIMILAR_IMPLEMENTATION_PATTERN.search(text):
            cls._append_intent(
                intents,
                "similar_implementation",
            )

        if TEST_INTENT_PATTERN.search(text):
            cls._append_intent(
                intents,
                "test_pattern",
            )

        if PROJECT_POLICY_PATTERN.search(text):
            cls._append_intent(
                intents,
                "project_policy",
            )

        return intents
