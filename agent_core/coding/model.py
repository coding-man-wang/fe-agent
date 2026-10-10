from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CodeFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str
    evidence: str = Field(
        min_length=1,
        description="从提供的代码上下文中引用的原文片段",
    )
    explanation: str = Field(
        min_length=1,
        description="基于该证据得到的分析结论",
    )


class ProposedCodeChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str
    objective: str = Field(min_length=1)
    proposed_approach: str = Field(min_length=1)


class ReadOnlyCodingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str

    summary: str = Field(min_length=1)

    findings: list[CodeFinding] = Field(default_factory=list)

    proposed_changes: list[ProposedCodeChange] = Field(default_factory=list)

    verification_plan: list[str] = Field(default_factory=list)

    missing_information: list[str] = Field(default_factory=list)

    execution_status: Literal["not_executed"] = "not_executed"
