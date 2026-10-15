from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class BenchmarkRetrievedFile(BaseModel):
    """
    Benchmark 视角下的一条 Retrieval 文件结果。

    这里不直接复用 Runtime 内部的
    HybridRelevantFile，避免 Benchmark
    与某一种 Retrieval 实现强耦合。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    relative_path: str

    category: str

    selection_score: float

    matched_queries: list[str] = Field(
        default_factory=list,
    )

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

    queries: list[str] = Field(
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

    mode: str = "lexical_baseline"

    indexed_files: int = Field(
        ge=0,
    )

    task_retrievals: list[BenchmarkTaskRetrieval] = Field(
        default_factory=list,
    )
