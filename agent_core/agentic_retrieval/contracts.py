"""Trusted task contract and model-facing retrieval/report schemas.

A criterion is fixed by the caller before retrieval. A model can report a gap,
not delete a criterion. Citation checks prove provenance, not entailment.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
PathText = Annotated[str, StringConstraints(min_length=1, max_length=300)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,80}$")]
Strategy = Literal["auto", "path", "symbol", "dependency", "exact", "hybrid"]
RunMode = Literal["fixed_context", "adaptive", "full_agentic"]
Role = Literal["candidate_target", "dependency", "reference", "test", "unknown"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Criterion(Contract):
    criterion_id: Identifier
    question: Text
    # Optional, caller-owned hard requirements; never populated from Benchmark Oracle.
    required_paths: tuple[PathText, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def unique_paths(self):
        if len(set(self.required_paths)) != len(self.required_paths):
            raise ValueError("duplicate required path")
        return self


class InvestigationScope(Contract):
    task_id: Identifier
    objective: Text
    criteria: tuple[Criterion, ...] = Field(min_length=1, max_length=24)
    non_goals: tuple[Text, ...] = Field(default=(), max_length=30)
    constraints: tuple[Text, ...] = Field(default=(), max_length=30)

    @model_validator(mode="after")
    def unique_criteria(self):
        ids = [c.criterion_id for c in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate criterion id")
        return self

    @classmethod
    def from_task(cls, task, requirement=None):
        if task.task_type != "investigation":
            raise ValueError("v1.6 only accepts investigation tasks")
        # Existing planner questions are the trusted starting contract, not
        # model-generated conclusions. For legacy plans use responsibilities,
        # then objective. Do not silently slice away questions.
        questions = (task.investigation_questions or task.responsibilities
                     or [task.objective or task.title])
        unique = list(dict.fromkeys(q.strip() for q in questions if q.strip()))
        if not unique:
            raise ValueError("no investigation criteria")
        return cls(task_id=task.task_id, objective=task.objective or task.title,
                   criteria=tuple(Criterion(criterion_id=f"C{i:02d}", question=q)
                                  for i, q in enumerate(unique, 1)),
                   non_goals=tuple(getattr(requirement, "non_goals", ())),
                   constraints=tuple(getattr(requirement, "constraints", ())))


class EvidenceGap(Contract):
    criterion_id: Identifier
    question: Text
    reason: Text
    strategy: Strategy = "auto"
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)]
    source_path: PathText | None = None

    @model_validator(mode="after")
    def validate_dependency(self):
        if self.strategy == "dependency" and not self.source_path:
            raise ValueError("dependency lookup requires source_path")
        return self


class Citation(Contract):
    evidence_id: Annotated[str, StringConstraints(pattern=r"^[SE][0-9]{3,6}$")]
    path: PathText
    line: int = Field(ge=1, strict=True)
    end_line: int | None = Field(default=None, ge=1, strict=True)
    quote: Annotated[str, StringConstraints(min_length=1, max_length=4000)]

    @model_validator(mode="after")
    def valid_range(self):
        end = self.end_line if self.end_line is not None else self.line
        if end < self.line or end - self.line >= 100 or not self.quote.strip():
            raise ValueError("invalid citation range or empty quote")
        return self


class CriterionAnswer(Contract):
    criterion_id: Identifier
    status: Literal["resolved", "unresolved"]
    claim: Text
    citations: tuple[Citation, ...] = Field(default=(), max_length=12)
    gap: EvidenceGap | None = None

    @model_validator(mode="after")
    def valid_status(self):
        if self.status == "resolved" and (not self.citations or self.gap is not None):
            raise ValueError("resolved answer requires citations and no gap")
        if self.status == "unresolved" and (self.gap is None or self.gap.criterion_id != self.criterion_id):
            raise ValueError("unresolved answer requires a gap for this criterion")
        return self


class FileAssessment(Contract):
    relative_path: PathText
    observed_role: Role
    rationale: Text
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    criterion_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=24)
    citations: tuple[Citation, ...] = Field(min_length=1, max_length=12)


class InvestigationReport(Contract):
    task_id: Identifier
    summary: Text
    answers: tuple[CriterionAnswer, ...] = Field(min_length=1, max_length=24)
    observed_files: tuple[FileAssessment, ...] = Field(default=(), max_length=60)
    limitations: tuple[Text, ...] = Field(default=(), max_length=30)


class RetrievalBudget(Contract):
    max_catalog_files: int = Field(default=1500, ge=1, le=10000)
    max_catalog_bytes: int = Field(default=12_000_000, ge=1, le=100_000_000)
    max_file_bytes: int = Field(default=256_000, ge=1, le=2_000_000)
    max_tree_bytes: int = Field(default=8_000_000, ge=1024, le=50_000_000)
    max_files: int = Field(default=12, ge=1, le=60)
    initial_max_files: int = Field(default=4, ge=1, le=60)
    max_candidates_per_call: int = Field(default=4, ge=1, le=20)
    max_seed_chars: int = Field(default=16000, ge=0, le=160000)
    max_evidence_chars: int = Field(default=40000, ge=1, le=200000)
    max_window_lines: int = Field(default=80, ge=1, le=100)
    max_no_progress_rounds: int = Field(default=2, ge=1, le=10)
    max_semantic_chunks: int = Field(default=2000, ge=1, le=10000)
    setup_timeout_seconds: float = Field(default=90, gt=0, le=300, allow_inf_nan=False)

    @model_validator(mode="after")
    def consistent(self):
        if self.initial_max_files > self.max_files:
            raise ValueError("initial file budget exceeds admission budget")
        if self.max_seed_chars > self.max_evidence_chars:
            raise ValueError("seed budget exceeds evidence budget")
        if self.max_file_bytes > self.max_catalog_bytes:
            raise ValueError("file budget exceeds catalog budget")
        return self
