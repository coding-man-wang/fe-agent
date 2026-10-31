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


ImportBindingKind = Literal[
    "named",
    "default",
    "namespace",
    "side_effect",
]


class ImportBinding(BaseModel):
    """
    一个 source file 中的
    Module Import Binding。

    示例：

        import {
            listOrders as fetchOrders
        } from "@/api/orders"

    imported_name:
        listOrders

    local_name:
        fetchOrders
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    source_path: str = Field(
        min_length=1,
    )

    module_specifier: str = Field(
        min_length=1,
    )

    binding_kind: ImportBindingKind

    imported_name: str | None = None

    local_name: str | None = None

    type_only: bool = False

    line_number: int = Field(
        ge=1,
    )


class ImportBindingExtractionResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    adapter_id: str = Field(
        min_length=1,
    )

    relative_path: str = Field(
        min_length=1,
    )

    bindings: list[ImportBinding] = Field(
        default_factory=list,
    )

    issues: list[CodeIntelligenceIssue] = Field(
        default_factory=list,
    )


DefinitionResolutionStatus = Literal[
    "resolved",
    "module_only",
    "external_module",
    "unresolved_module",
    "symbol_not_found",
    "unsupported",
    "export_not_found",
    "ambiguous_export",
    "cyclic_export",
]


class SymbolDefinitionResolution(BaseModel):
    """
    Import Binding 的 Definition Resolution。

    注意：

    Resolution 只是代码理解结果。

    它不是：
        Change Surface
        Write Permission
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    source_path: str

    local_name: str | None = None

    imported_name: str | None = None

    module_specifier: str

    status: DefinitionResolutionStatus

    target_path: str | None = None

    target_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    reason: str | None = None

    module_path: str | None = None

    resolution_chain: list[str] = Field(
        default_factory=list,
    )


ModuleExportBindingKind = Literal[
    "local",
    "reexport",
    "star_reexport",
]


class ModuleExportBinding(BaseModel):
    """
    Module 对外暴露的一个 Export Binding。

    示例：

        export { listOrders as fetchOrders }

    exported_name:
        fetchOrders

    local_name:
        listOrders


        export {
            listOrders as fetchOrders
        } from "./orders"

    exported_name:
        fetchOrders

    imported_name:
        listOrders

    module_specifier:
        ./orders
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    source_path: str = Field(
        min_length=1,
    )

    binding_kind: ModuleExportBindingKind

    exported_name: str | None = None

    local_name: str | None = None

    module_specifier: str | None = None

    imported_name: str | None = None

    type_only: bool = False

    line_number: int = Field(
        ge=1,
    )

    # Vue Component implicit default export
    # 等 Adapter 生成的语义可标记 synthetic。
    synthetic: bool = False

    @model_validator(mode="after")
    def validate_binding(
        self,
    ) -> "ModuleExportBinding":

        if self.binding_kind == "local":

            if not self.exported_name:
                raise ValueError("local export 必须提供 " "exported_name")

        elif self.binding_kind == "reexport":

            if not self.exported_name:
                raise ValueError("reexport 必须提供 " "exported_name")

            if not self.imported_name:
                raise ValueError("reexport 必须提供 " "imported_name")

            if not self.module_specifier:
                raise ValueError("reexport 必须提供 " "module_specifier")

        elif self.binding_kind == "star_reexport":

            if not self.module_specifier:
                raise ValueError("star_reexport 必须提供 " "module_specifier")

        return self


class ModuleExportExtractionResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    adapter_id: str

    relative_path: str

    exports: list[ModuleExportBinding] = Field(
        default_factory=list,
    )

    issues: list[CodeIntelligenceIssue] = Field(
        default_factory=list,
    )


ReferenceKind = Literal[
    "call",
    "read",
    "type_reference",
]


ReferenceTargetForm = Literal[
    "identifier",
    "member",
    "unknown",
]


class SymbolReferenceOccurrence(BaseModel):
    """
    Source Code 中出现的一次 Symbol Reference。

    本阶段只真正解析：

        reference_kind = call
        target_form = identifier

    但 Contract 提前允许：
        read
        type_reference

    下一阶段直接复用。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    source_path: str = Field(
        min_length=1,
    )

    reference_kind: ReferenceKind

    target_form: ReferenceTargetForm

    # 对：
    #
    # listOrders()
    #
    # name = listOrders
    #
    # 对：
    #
    # api.listOrders()
    #
    # 4A 中 name 可以为空，
    # expression_text 保存完整表达式。
    name: str | None = None

    expression_text: str = Field(
        min_length=1,
    )

    receiver_name: str | None = None

    member_name: str | None = None

    line_number: int = Field(
        ge=1,
    )

    column_number: int = Field(
        ge=1,
    )


class SymbolReferenceExtractionResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    adapter_id: str

    relative_path: str

    references: list[SymbolReferenceOccurrence] = Field(
        default_factory=list,
    )

    issues: list[CodeIntelligenceIssue] = Field(
        default_factory=list,
    )


SymbolReferenceResolutionStatus = Literal[
    "resolved",
    "external",
    "unresolved",
    "ambiguous",
    "unsupported",
]


class ResolvedSymbolReference(BaseModel):
    """
    一次 Reference 解析后的结果。

    source_symbol_id:
        谁在引用 / 调用

    target_symbol_ids:
        引用了谁
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    source_path: str

    reference_kind: ReferenceKind

    expression_text: str

    line_number: int = Field(
        ge=1,
    )

    source_symbol_id: str | None = None

    status: SymbolReferenceResolutionStatus

    target_symbol_ids: list[str] = Field(
        default_factory=list,
    )

    reason: str | None = None


SymbolEdgeRelation = Literal[
    "calls",
    "references",
]


class SymbolGraphEdge(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    source_symbol_id: str = Field(
        min_length=1,
    )

    target_symbol_id: str = Field(
        min_length=1,
    )

    relation: SymbolEdgeRelation

    source_path: str

    line_number: int = Field(
        ge=1,
    )
