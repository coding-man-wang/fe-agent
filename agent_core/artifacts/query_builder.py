import re

from agent_core.codebase.retrieval_query import RetrievalQuery, RetrievalQueryBuilder
from agent_core.sdd.model import RequirementSpec, TaskDefinition


STRUCTURAL_INPUT_PATTERNS = (
    re.compile(r"^RequirementSpec(?:\b|$)", re.IGNORECASE),
    re.compile(r"^TechnicalSpec(?:\b|$)", re.IGNORECASE),
    re.compile(r"^ImplementationPlan\s+step\s+\d+", re.IGNORECASE),
    re.compile(r"^ProjectContext(?:\b|$)", re.IGNORECASE),
    re.compile(r"^TASK-\d+\s*(?:输出|output)", re.IGNORECASE),
)


class ArtifactAwareRetrievalQueryBuilder:
    """
    保留 v0.4 QueryBuilder 的检索策略，只过滤“结构化工作流引用”。

    这些 input 应由 Artifact/Spec 数据流消费，而不是拿去搜索代码仓：
    - ImplementationPlan step N
    - TASK-001 输出：...
    - RequirementSpec / TechnicalSpec / ProjectContext 引用
    """

    def __init__(
        self,
        *,
        max_queries: int = 6,
        inner: RetrievalQueryBuilder | None = None,
    ) -> None:
        self.inner = inner or RetrievalQueryBuilder(max_queries=max_queries)

    def build(
        self,
        *,
        task: TaskDefinition,
        raw_requirement: str | None = None,
        requirement_spec: RequirementSpec | None = None,
    ) -> list[RetrievalQuery]:
        queries = self.inner.build(
            task=task,
            raw_requirement=raw_requirement,
            requirement_spec=requirement_spec,
        )
        filtered = [query for query in queries if not self._is_structural_input(query)]
        return filtered or queries

    @staticmethod
    def _is_structural_input(query: RetrievalQuery) -> bool:
        if "input" not in query.kinds:
            return False
        text = query.text.strip()
        return any(pattern.search(text) for pattern in STRUCTURAL_INPUT_PATTERNS)
