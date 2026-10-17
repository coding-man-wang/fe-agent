from typing import Literal

from pydantic import BaseModel, Field


class RepositoryProfile(BaseModel):
    project_name: str = Field(description="项目名称")

    root_path: str = Field(description="代码仓库根目录")

    framework: str = Field(
        default="Unknown",
        description="前端框架，例如 Vue3、React、Next.js",
    )

    language: str = Field(
        default="JavaScript",
        description="项目主要语言，例如 TypeScript",
    )

    build_tool: str | None = Field(
        default=None,
        description="构建工具，例如 Vite、Webpack",
    )

    package_manager: str | None = Field(
        default=None,
        description="包管理工具，例如 pnpm、npm、yarn",
    )

    state_management: str | None = Field(
        default=None,
        description="状态管理方案，例如 Pinia、Redux",
    )

    test_stack: list[str] = Field(
        default_factory=list,
        description="测试相关技术栈",
    )

    scripts: dict[str, str] = Field(
        default_factory=dict,
        description="package.json 中定义的 scripts",
    )

    dependencies: list[str] = Field(
        default_factory=list,
        description="项目使用的 npm 依赖名称",
    )

    config_files: list[str] = Field(
        default_factory=list,
        description="识别到的重要项目配置文件",
    )


FileCategory = Literal[
    "source",
    "test",
    "config",
    "documentation",
    "asset",
    "other",
]


class FileRecord(BaseModel):
    relative_path: str = Field(description="相对于代码仓库根目录的文件路径")

    name: str = Field(description="文件名")

    suffix: str = Field(
        default="",
        description="文件扩展名",
    )

    size_bytes: int = Field(
        ge=0,
        description="文件大小",
    )

    category: FileCategory = Field(description="文件分类")

    language: str | None = Field(
        default=None,
        description="能够识别时记录主要语言",
    )

    is_indexable: bool = Field(description="后续是否允许进入文本检索和 Context")

    skip_reason: str | None = Field(
        default=None,
        description="不可索引时记录原因",
    )


class RepositoryFileIndex(BaseModel):
    root_path: str

    total_files: int = Field(
        ge=0,
    )

    indexable_files: int = Field(
        ge=0,
    )

    files: list[FileRecord] = Field(
        default_factory=list,
    )

    category_counts: dict[str, int] = Field(
        default_factory=dict,
    )


SearchMatchType = Literal[
    "filename",
    "path",
    "content",
]


class ExactSearchMatch(BaseModel):
    match_type: SearchMatchType

    line_number: int | None = Field(
        default=None,
        ge=1,
        description="内容匹配时对应的代码行号",
    )

    snippet: str | None = Field(
        default=None,
        description="匹配位置附近的文本片段",
    )


class ExactSearchResult(BaseModel):
    relative_path: str = Field(description="匹配文件相对于仓库根目录的路径")

    category: str = Field(description="文件分类")

    language: str | None = Field(
        default=None,
        description="文件主要语言",
    )

    score: int = Field(
        ge=0,
        description="当前结果的确定性相关性评分",
    )

    matches: list[ExactSearchMatch] = Field(
        default_factory=list,
        description="该文件中所有匹配信息",
    )


class ExactSearchResponse(BaseModel):
    query: str

    scanned_files: int = Field(
        ge=0,
    )

    matched_files: int = Field(
        ge=0,
    )

    results: list[ExactSearchResult] = Field(
        default_factory=list,
    )


class RelevantFile(BaseModel):
    relative_path: str = Field(description="相关文件路径")

    category: str = Field(description="文件分类")

    language: str | None = None

    size_bytes: int = Field(
        ge=0,
    )

    score: int = Field(
        ge=0,
        description="综合相关性分数",
    )

    matched_queries: list[str] = Field(
        default_factory=list,
        description="哪些搜索关键词命中了当前文件",
    )

    reasons: list[str] = Field(
        default_factory=list,
        description="为什么选择当前文件",
    )


class RelevantFileSelection(BaseModel):
    candidate_count: int = Field(
        ge=0,
    )

    selected_count: int = Field(
        ge=0,
    )

    total_selected_bytes: int = Field(
        ge=0,
    )

    selected_files: list[RelevantFile] = Field(
        default_factory=list,
    )

    skipped_by_budget: list[str] = Field(
        default_factory=list,
    )


class CodeChunk(BaseModel):
    chunk_id: str = Field(description="Chunk 唯一 ID")

    relative_path: str = Field(description="来源文件")

    start_line: int = Field(
        ge=1,
        description="起始行",
    )

    end_line: int = Field(
        ge=1,
        description="结束行",
    )

    content: str = Field(description="Chunk 的代码内容")

    language: str | None = None

    category: str = "source"


class SemanticSearchResult(BaseModel):
    chunk: CodeChunk

    score: float = Field(description="向量相似度得分")


class SemanticSearchResponse(BaseModel):
    query: str

    results: list[SemanticSearchResult] = Field(
        default_factory=list,
    )


class HybridFileCandidate(BaseModel):
    relative_path: str

    lexical_rank: int | None = Field(
        default=None,
        ge=1,
    )

    semantic_rank: int | None = Field(
        default=None,
        ge=1,
    )

    lexical_score: int | None = Field(
        default=None,
        ge=0,
    )

    semantic_score: float | None = None

    best_semantic_chunk: CodeChunk | None = None

    fusion_score: float = Field(
        ge=0,
    )

    reasons: list[str] = Field(
        default_factory=list,
    )


class HybridSearchResponse(BaseModel):
    query: str

    candidates: list[HybridFileCandidate] = Field(
        default_factory=list,
    )


ContextRole = Literal[
    "implementation_target",
    "dependency",
    "reference",
    "test",
    "policy",
    "unclassified",
]


TargetMatchType = Literal[
    "exact_file",
    "symbol",
    "directory",
    "none",
]


class HybridRelevantFile(BaseModel):
    relative_path: str = Field(description="相关文件路径")

    category: str = Field(description="文件分类")

    language: str | None = None

    size_bytes: int = Field(
        ge=0,
    )

    retrieval_score: float = Field(
        ge=0,
        description="多个 Hybrid Query 聚合后的检索相关性分数",
    )

    retrieval_rank: int = Field(
        ge=1,
        description="根据 Retrieval Evidence 得到的排名",
    )

    policy_score: int = Field(
        ge=0,
        description="根据 Task / Module 等 Context Policy 得到的分数",
    )

    policy_rank: int | None = Field(
        default=None,
        ge=1,
    )

    selection_score: float = Field(
        ge=0,
        description="最终 Selection Rank Fusion 分数",
    )

    best_semantic_chunk: CodeChunk | None = None

    best_semantic_score: float | None = None

    query_hit_count: int = Field(
        default=0,
        ge=0,
        description="当前文件被多少个不同 Query 命中",
    )

    lexical_hit_count: int = Field(
        default=0,
        ge=0,
        description=("当前文件被 Lexical Retrieval " "命中的 Query 数量"),
    )

    semantic_hit_count: int = Field(
        default=0,
        ge=0,
        description=("当前文件被 Semantic Retrieval " "命中的 Query 数量"),
    )

    hybrid_hit_count: int = Field(
        default=0,
        ge=0,
        description=("同一个 Query 下同时被 " "Lexical 和 Semantic 命中的数量"),
    )

    matched_queries: list[str] = Field(
        default_factory=list,
        description="哪些搜索关键词命中了当前文件",
    )

    matched_intents: list[str] = Field(
        default_factory=list,
        description=("命中当前文件的 Retrieval Query " "包含哪些搜索意图"),
    )

    # =====================================
    # Target Evidence
    #
    # target_modules 是 Planner 对
    # Change Surface 的假设。
    #
    # 这里把文件和 target_modules
    # 之间的匹配关系结构化保存。
    # =====================================

    target_match_type: TargetMatchType = Field(
        default="none",
        description=("当前文件与 Task target_modules " "之间的匹配类型"),
    )

    target_match_value: str | None = Field(
        default=None,
        description=("当前文件实际命中的 " "target_modules 原始值"),
    )

    context_role: ContextRole = Field(
        default="unclassified",
        description=("当前文件在本次 Task Context " "中的用途"),
    )

    role_reason: str | None = Field(
        default=None,
        description="为什么把当前文件分类为该角色",
    )

    rerank_score: float | None = None

    rerank_rank: int | None = Field(
        default=None,
        ge=1,
    )

    reasons: list[str] = Field(
        default_factory=list,
        description="为什么选择当前文件",
    )


class HybridRelevantFileSelection(BaseModel):
    candidate_count: int = Field(
        ge=0,
    )

    eligible_count: int = Field(
        ge=0,
    )

    selected_count: int = Field(
        ge=0,
    )

    total_selected_bytes: int = Field(
        ge=0,
    )

    selected_files: list[HybridRelevantFile] = Field(
        default_factory=list,
    )

    skipped_by_budget: list[str] = Field(
        default_factory=list,
    )

    skipped_non_indexable: list[str] = Field(
        default_factory=list,
    )
