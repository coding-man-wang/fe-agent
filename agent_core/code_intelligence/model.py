from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

# ==================================
# Technology Profile
#
# 这里只保存 canonical technology ID。
#
# Agent Core 不应该直接依赖：
#
# Vue3
# React
# Vite
# Vitest
#
# 这些具体字符串的判断逻辑。
# ==================================


class TechnologyProfile(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    languages: list[str] = Field(
        default_factory=list,
    )

    frameworks: list[str] = Field(
        default_factory=list,
    )

    build_tools: list[str] = Field(
        default_factory=list,
    )

    test_frameworks: list[str] = Field(
        default_factory=list,
    )

    package_managers: list[str] = Field(
        default_factory=list,
    )

    state_management: list[str] = Field(
        default_factory=list,
    )

    def has(
        self,
        technology_id: str,
    ) -> bool:
        """
        查询 Repository 是否具备某种
        Technology Capability Source。

        例如：

            profile.has("vue")
            profile.has("typescript")

        Core Module 不应该通过：
            framework == "Vue3"

        判断具体框架。
        """

        technology_id = technology_id.strip().lower()

        return technology_id in {
            *self.languages,
            *self.frameworks,
            *self.build_tools,
            *self.test_frameworks,
            *self.package_managers,
            *self.state_management,
        }


# ==================================
# Source Document
#
# Code Intelligence 的最小输入。
#
# 不直接传 RepositoryProfile，
# 防止 Parser / Symbol Core
# 和现有 Repository Scanner 耦合。
# ==================================


class SourceDocument(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    relative_path: str = Field(
        min_length=1,
    )

    content: str

    language: str | None = None


# ==================================
# Common Symbol Model
#
# 这是 Framework-independent
# Code Intelligence Contract。
#
# Vue / React / TS Adapter
# 最终都必须转换成该模型。
# ==================================


SymbolKind = Literal[
    "module",
    "function",
    "class",
    "method",
    "variable",
    "type",
    "interface",
    "component",
    "unknown",
]


class CodeSymbol(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    symbol_id: str = Field(
        min_length=1,
    )

    name: str = Field(
        min_length=1,
    )

    kind: SymbolKind

    relative_path: str = Field(
        min_length=1,
    )

    start_line: int = Field(
        ge=1,
    )

    end_line: int = Field(
        ge=1,
    )

    exported: bool = False

    parent_symbol_id: str | None = None

    @model_validator(mode="after")
    def validate_range(
        self,
    ) -> "CodeSymbol":

        if self.end_line < self.start_line:
            raise ValueError("CodeSymbol end_line " "不能小于 start_line")

        return self


# ==================================
# Extraction Diagnostic
#
# AST Syntax Error /
# Unsupported Syntax
# 不应该直接炸毁整个 Repository。
# ==================================


CodeIntelligenceIssueSeverity = Literal[
    "warning",
    "error",
]


class CodeIntelligenceIssue(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    severity: CodeIntelligenceIssueSeverity

    message: str = Field(
        min_length=1,
    )

    line_number: int | None = Field(
        default=None,
        ge=1,
    )


class SymbolExtractionResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    adapter_id: str = Field(
        min_length=1,
    )

    relative_path: str = Field(
        min_length=1,
    )

    symbols: list[CodeSymbol] = Field(
        default_factory=list,
    )

    issues: list[CodeIntelligenceIssue] = Field(
        default_factory=list,
    )
