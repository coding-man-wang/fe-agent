from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_core.codebase.model import RepositoryProfile
from agent_core.sdd.model import RequirementSpec, TaskDefinition, TechnicalSpec

ContextContentMode = Literal["full_file", "semantic_chunk", "truncated"]


class CodeContextItem(BaseModel):
    relative_path: str
    category: str
    language: str | None = None
    content_mode: ContextContentMode
    content: str
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    relevance_reasons: list[str] = Field(default_factory=list)
    content_chars: int = Field(ge=0)


class ContextBudgetConfig(BaseModel):
    max_files: int = Field(default=6, ge=1)
    max_total_chars: int = Field(default=30_000, ge=1)
    max_chars_per_file: int = Field(default=8_000, ge=1)


class ContextBudgetUsage(BaseModel):
    max_files: int = Field(ge=1)
    max_total_chars: int = Field(ge=1)
    max_chars_per_file: int = Field(ge=1)
    used_files: int = Field(ge=0)
    used_chars: int = Field(ge=0)
    skipped_files: list[str] = Field(default_factory=list)
    compressed_files: list[str] = Field(default_factory=list)


class TaskContextBundle(BaseModel):
    requirement: RequirementSpec
    technical_spec: TechnicalSpec
    task: TaskDefinition
    repository: RepositoryProfile
    engineering_rules: list[str] = Field(default_factory=list)
    code_context: list[CodeContextItem] = Field(default_factory=list)
    budget: ContextBudgetUsage

    # v1.0: 下游 Task 真正消费上游 Artifact，而不是把“TASK-001 输出”当搜索词。
    upstream_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    change_surface: dict[str, Any] | None = None


class RenderedContext(BaseModel):
    system_message: str = Field(description="系统级行为指令")
    user_message: str = Field(description="当前任务及上下文数据")
