"""Immutable evidence windows and fail-closed completion gate."""
from __future__ import annotations

import re
from typing import Iterable

from pydantic import Field, ValidationError, model_validator

from .catalog import CommittedCatalog
from .contracts import Citation, Contract, InvestigationReport, InvestigationScope


class EvidenceError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class EvidenceWindow(Contract):
    evidence_id: str
    path: str
    revision: str
    sha256: str
    start_line: int = Field(ge=1, strict=True)
    lines: tuple[str, ...] = Field(min_length=1, max_length=100)

    @property
    def end_line(self):
        return self.start_line + len(self.lines) - 1

    def projection(self):
        return {"evidence_id": self.evidence_id, "path": self.path,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "numbered_lines": [{"line": i, "content": text}
                                   for i, text in enumerate(self.lines, self.start_line)]}

    @classmethod
    def from_catalog(cls, catalog: CommittedCatalog, evidence_id: str, path: str,
                     start: int, end: int):
        if path not in catalog.files or not (1 <= start <= end and end - start < 100):
            raise EvidenceError("invalid_window", "invalid evidence path/range")
        lines = tuple(catalog.files[path].splitlines()[start - 1:end])
        if not lines:
            raise EvidenceError("invalid_window", "empty evidence range")
        return cls(evidence_id=evidence_id, path=path, revision=catalog.revision,
                   sha256=catalog.digest(path), start_line=start, lines=lines)


def parse_report(content: str) -> InvestigationReport:
    # Unwrap only a complete fenced JSON document; never repair content.
    content = content.strip()
    match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", content, re.I)
    if match:
        content = match.group(1).strip()
    try:
        return InvestigationReport.model_validate_json(content)
    except ValidationError as exc:
        raise EvidenceError("invalid_report_schema", "report does not match schema") from exc


class EvidenceLedger:
    def __init__(self, catalog: CommittedCatalog, windows: Iterable[EvidenceWindow] = ()):
        self.catalog = catalog
        self._windows: dict[str, EvidenceWindow] = {}
        for window in windows:
            self.add(window)

    @property
    def windows(self):
        return tuple(self._windows.values())

    def add(self, window: EvidenceWindow):
        if window.evidence_id in self._windows:
            raise EvidenceError("duplicate_evidence_id", "duplicate evidence reference")
        expected = EvidenceWindow.from_catalog(self.catalog, window.evidence_id,
                                               window.path, window.start_line, window.end_line)
        if expected != window or not re.fullmatch(r"[SE][0-9]{3,6}", window.evidence_id):
            raise EvidenceError("snapshot_evidence_mismatch", "evidence differs from committed source")
        self._windows[window.evidence_id] = window

    @classmethod
    def from_observations(cls, catalog: CommittedCatalog, seeds, observations):
        ledger = cls(catalog, seeds)
        seen_call_ids = set()
        for obs in observations:
            if obs.call_id in seen_call_ids:
                raise EvidenceError("duplicate_call_id", "duplicated runtime observation")
            seen_call_ids.add(obs.call_id)
            if not obs.success or obs.tool_name != "read_file":
                continue
            try:
                data = obs.data
                if not isinstance(obs.evidence_id, str) or not re.fullmatch(r"E[0-9]{3,6}", obs.evidence_id):
                    raise ValueError("missing runtime evidence id")
                window = EvidenceWindow(evidence_id=obs.evidence_id, path=data["path"],
                                        revision=data["revision"], sha256=data["sha256"],
                                        start_line=data["start_line"], lines=tuple(data["lines"]))
                if window.end_line != data["end_line"]:
                    raise ValueError("line count mismatch")
                expected_numbered = [{"line": i, "content": text}
                                     for i, text in enumerate(window.lines, window.start_line)]
                if data.get("numbered_lines") != expected_numbered:
                    raise ValueError("numbered lines mismatch")
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise EvidenceError("invalid_observation", "malformed successful read") from exc
            ledger.add(window)
        return ledger

    def validate(self, citation: Citation):
        window = self._windows.get(citation.evidence_id)
        end = citation.end_line if citation.end_line is not None else citation.line
        if (window is None or window.path != citation.path
                or not (window.start_line <= citation.line <= end <= window.end_line)):
            raise EvidenceError("invalid_citation", "reference outside received evidence")
        source_lines = window.lines[citation.line - window.start_line:end - window.start_line + 1]
        # A range cannot claim 10 lines of coverage using a one-line quote.
        # Interior lines must match exactly, with partial first/last lines allowed.
        quote_lines = citation.quote.split("\n")
        if len(quote_lines) != len(source_lines):
            raise EvidenceError("citation_text_mismatch", "quote/range length mismatch")
        if len(source_lines) == 1:
            valid = citation.quote in source_lines[0]
        else:
            valid = (source_lines[0].endswith(quote_lines[0])
                     and source_lines[-1].startswith(quote_lines[-1])
                     and tuple(quote_lines[1:-1]) == source_lines[1:-1]
                     and bool(quote_lines[0].strip()) and bool(quote_lines[-1].strip()))
        if not valid:
            raise EvidenceError("citation_text_mismatch", "quote does not match exact source range")
        return window


class CompletionGate:
    def __init__(self, scope: InvestigationScope, catalog: CommittedCatalog,
                 seeds: tuple[EvidenceWindow, ...], *, require_roles: bool = False,
                 candidate_target_ceiling: frozenset[str] | None = None):
        self.scope = scope
        self.catalog = catalog
        self.seeds = seeds
        self.require_roles = require_roles
        self.candidate_target_ceiling = candidate_target_ceiling

    def validate(self, content: str, observations) -> InvestigationReport:
        report = parse_report(content)
        if report.task_id != self.scope.task_id:
            raise EvidenceError("report_identity_mismatch", "task mismatch")
        ledger = EvidenceLedger.from_observations(self.catalog, self.seeds, observations)
        criteria = {item.criterion_id: item for item in self.scope.criteria}
        answers = {item.criterion_id: item for item in report.answers}
        if len(answers) != len(report.answers) or answers.keys() != criteria.keys():
            raise EvidenceError("criterion_coverage_mismatch", "missing, extra or duplicate criterion")
        answer_paths: dict[str, set[str]] = {}
        unresolved = []
        for criterion_id, answer in answers.items():
            for citation in answer.citations:
                ledger.validate(citation)
            paths = {c.path for c in answer.citations}
            answer_paths[criterion_id] = paths
            if answer.status == "unresolved":
                unresolved.append(criterion_id)
            elif not set(criteria[criterion_id].required_paths) <= paths:
                raise EvidenceError("required_evidence_missing", "criterion lacks required path evidence")
        role_paths = set()
        for assessment in report.observed_files:
            path = assessment.relative_path
            if path in role_paths:
                raise EvidenceError("duplicate_role_path", "duplicate file assessment")
            role_paths.add(path)
            ids = set(assessment.criterion_ids)
            if len(ids) != len(assessment.criterion_ids) or not ids <= criteria.keys():
                raise EvidenceError("invalid_role_criterion", "file role references unknown criterion")
            for citation in assessment.citations:
                ledger.validate(citation)
            if any(c.path != path for c in assessment.citations):
                raise EvidenceError("invalid_role_evidence", "role needs evidence from its own file")
            if not any(path in answer_paths[cid] for cid in ids):
                raise EvidenceError("invalid_role_evidence", "role is not tied to a criterion answer")
            if assessment.observed_role in {"candidate_target", "test"}:
                if self.candidate_target_ceiling is not None and path not in self.candidate_target_ceiling:
                    raise EvidenceError("new_target_requires_approval", "new write-bearing role requires host approval")
                if not any(answers[cid].status == "resolved" for cid in ids):
                    raise EvidenceError("unresolved_target", "candidate target has no resolved criterion")
        if unresolved:
            raise EvidenceError("unresolved_criteria", "one or more mandatory criteria unresolved")
        if self.require_roles and not set().union(*answer_paths.values()) <= role_paths:
            raise EvidenceError("missing_file_assessment", "every cited file needs a role assessment")
        return report

    def __call__(self, content: str, observations) -> None:
        self.validate(content, observations)
