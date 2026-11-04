from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

from agent_core.codebase.model import (
    RepositoryProfile,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)

ContextContentMode = Literal[
    "full_file",
    "semantic_chunk",
    "symbol_window",
    "truncated",
]


ContextSegmentSource = Literal[
    "symbol",
    "local_context",
    "container_evidence",
    "semantic_evidence",
    "import_coverage",
]


class CodeContextSegment(BaseModel):
    """
    一段来自真实 Repository File 的源码范围。

    Symbol Window 允许一个 CodeContextItem
    包含多个不连续 Segment。

    例如：

        lines 1-3     imports
        lines 120-145 target function

    注意：
        Segment 是 Source Range，
        不是 Virtual Document Range。
    """

    start_line: int = Field(
        ge=1,
    )

    end_line: int = Field(
        ge=1,
    )

    sources: list[ContextSegmentSource] = Field(
        default_factory=list,
    )

    symbol_ids: list[str] = Field(
        default_factory=list,
    )

    reasons: list[str] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_range(
        self,
    ) -> "CodeContextSegment":

        if self.end_line < self.start_line:
            raise ValueError("CodeContextSegment " "end_line 不能小于 start_line")

        return self


class CodeContextItem(BaseModel):
    relative_path: str

    category: str

    language: str | None = None

    content_mode: ContextContentMode

    # ==================================
    # Canonical Context Text
    #
    # ReadOnlyCodingAgent 仍然从这里
    # 校验 findings.evidence。
    #
    # 因此 v1.4.6 不改变这个 Contract。
    # ==================================

    content: str

    # full_file / semantic_chunk /
    # truncated 可以表示连续范围。
    #
    # symbol_window 可能是多个不连续范围，
    # 因此 start/end 为 None，
    # 真实位置放在 segments。
    start_line: int | None = Field(
        default=None,
        ge=1,
    )

    end_line: int | None = Field(
        default=None,
        ge=1,
    )

    relevance_reasons: list[str] = Field(
        default_factory=list,
    )

    # 仅用于解释阅读相关性。
    #
    # 不转换成：
    # - Change Surface
    # - Write Permission
    symbol_candidates: list[SymbolCandidate] = Field(
        default_factory=list,
    )

    # ==================================
    # v1.4.6 Symbol Window
    # ==================================

    segments: list[CodeContextSegment] = Field(
        default_factory=list,
    )

    # 完整 Symbol Source Range
    # 已经进入 Context。
    covered_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    # Container Symbol，例如：
    #
    # Vue Component 1-500
    #
    # 不应该为了一个内部相关行为
    # 自动发送整个 Component。
    #
    # localized 表示：
    #
    # Container 已经通过更窄的
    # Repository Evidence 定位。
    localized_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    # Symbol-aware Context 不允许
    # 静默丢失 Mandatory Evidence。
    missing_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    window_issues: list[str] = Field(
        default_factory=list,
    )

    content_chars: int = Field(
        ge=0,
    )

    # Context Compression 前
    # 原文件字符数。
    source_file_chars: int = Field(
        default=0,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_symbol_window(
        self,
    ) -> "CodeContextItem":

        if self.content_mode == "symbol_window":
            if not self.segments:
                raise ValueError(
                    "symbol_window 必须包含 " "至少一个真实 Source Segment"
                )

            if self.start_line is not None or self.end_line is not None:
                raise ValueError(
                    "symbol_window 可能由多个"
                    "不连续源码段组成，"
                    "不能伪装成一个连续 "
                    "start_line/end_line 范围"
                )

        return self


class ContextBudgetConfig(BaseModel):
    max_files: int = Field(
        default=6,
        ge=1,
    )

    max_total_chars: int = Field(
        default=30_000,
        ge=1,
    )

    max_chars_per_file: int = Field(
        default=8_000,
        ge=1,
    )


class ContextBudgetUsage(BaseModel):
    max_files: int = Field(
        ge=1,
    )

    max_total_chars: int = Field(
        ge=1,
    )

    max_chars_per_file: int = Field(
        ge=1,
    )

    used_files: int = Field(
        ge=0,
    )

    used_chars: int = Field(
        ge=0,
    )

    skipped_files: list[str] = Field(
        default_factory=list,
    )

    compressed_files: list[str] = Field(
        default_factory=list,
    )

    # ==================================
    # v1.4.6 Diagnostics
    # ==================================

    symbol_window_files: list[str] = Field(
        default_factory=list,
    )

    symbol_window_saved_chars: int = Field(
        default=0,
        ge=0,
    )


class TaskContextBundle(BaseModel):
    requirement: RequirementSpec

    technical_spec: TechnicalSpec

    task: TaskDefinition

    repository: RepositoryProfile

    engineering_rules: list[str] = Field(
        default_factory=list,
    )

    code_context: list[CodeContextItem] = Field(
        default_factory=list,
    )

    budget: ContextBudgetUsage

    upstream_artifacts: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    change_surface: dict[str, Any] | None = None

    # ==================================
    # v1.3D
    #
    # Harness-selected
    # Verification Strategy。
    #
    # 使用 dict 避免：
    #
    # context
    #    ↓
    # benchmark
    #
    # 的反向依赖。
    # ==================================

    verification_strategy: dict[str, Any] | None = None


class RenderedContext(BaseModel):
    system_message: str = Field(description="系统级行为指令")

    user_message: str = Field(description="当前任务及上下文数据")
