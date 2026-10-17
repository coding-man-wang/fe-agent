from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class BenchmarkRetrievalQuery(BaseModel):
    """
    Benchmark 记录的一条搜索 Query。

    不只是记录搜索了什么，
    还记录 Query 来源和 Retrieval Intent。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    text: str

    sources: list[str] = Field(
        default_factory=list,
    )

    kinds: list[str] = Field(
        default_factory=list,
    )

    intents: list[str] = Field(
        default_factory=list,
    )


class BenchmarkRetrievedFile(BaseModel):
    """
    Benchmark 视角下的一条
    Retrieval 文件结果。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    relative_path: str

    category: str

    selection_score: float

    retrieval_score: float = 0.0

    best_semantic_score: float | None = None

    query_hit_count: int = 0

    lexical_hit_count: int = 0

    semantic_hit_count: int = 0

    hybrid_hit_count: int = 0

    matched_queries: list[str] = Field(
        default_factory=list,
    )

    matched_intents: list[str] = Field(
        default_factory=list,
    )

    # =====================================
    # Target Evidence
    # =====================================

    target_match_type: str = "none"

    target_match_value: str | None = None

    context_role: str = "unclassified"

    role_reason: str | None = None

    rerank_score: float | None = None

    rerank_rank: int | None = None

    reasons: list[str] = Field(
        default_factory=list,
    )


class BenchmarkTaskRetrieval(BaseModel):
    """
    单个 Task 的 Retrieval 结果。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    task_id: str

    task_type: str

    title: str

    target_modules: list[str] = Field(
        default_factory=list,
    )

    inputs: list[str] = Field(
        default_factory=list,
    )

    depends_on: list[str] = Field(
        default_factory=list,
    )

    # -----------------------------------
    # 兼容旧结构。
    #
    # 报表如果只想看到搜索词，
    # 继续读 queries 即可。
    # -----------------------------------

    queries: list[str] = Field(
        default_factory=list,
    )

    # -----------------------------------
    # 带来源 / Kind / Intent 的 Query。
    # -----------------------------------

    query_details: list[BenchmarkRetrievalQuery] = Field(
        default_factory=list,
    )

    selected_files: list[BenchmarkRetrievedFile] = Field(
        default_factory=list,
    )

    candidate_count: int = Field(
        default=0,
        ge=0,
    )

    selected_count: int = Field(
        default=0,
        ge=0,
    )


class BenchmarkRetrievalCapture(BaseModel):
    """
    一次 Benchmark Run 中
    整个 TaskPlan 的 Retrieval Capture。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    mode: str = "lexical_requirement_aware_v0.2"

    indexed_files: int = Field(
        ge=0,
    )

    task_retrievals: list[BenchmarkTaskRetrieval] = Field(
        default_factory=list,
    )
