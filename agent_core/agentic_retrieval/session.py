"""Per-run candidate admission, bounded source reads, and progress tracking."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import Field, model_validator

from agent_core.runtime.model import Observation, RuntimeConfig
from agent_core.tools.base import Tool
from agent_core.tools.registry import ToolRegistry

from .contracts import Contract, EvidenceGap, InvestigationScope, PathText, RetrievalBudget, RetrievalRequest
from .evidence import EvidenceWindow
from .retrieval import RetrievalBackend


class ReadWindowArgs(Contract):
    path: PathText
    start_line: int = Field(default=1, ge=1, strict=True)
    end_line: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def check_range(self):
        if self.end_line is not None and not self.start_line <= self.end_line < self.start_line + 100:
            raise ValueError("read at most 100 lines")
        return self


def _fits(data: dict, tool_name: str, config: RuntimeConfig) -> bool:
    # The v1.5 dispatcher checks serialized Observation bytes BEFORE compact
    # evidence projection. Bound both code copies and worst-case 256-char ID.
    worst_case = Observation(call_id="\x01" * 256, tool_name=tool_name,
                             success=True, data=data)
    return len(worst_case.output().encode("utf-8")) <= config.max_tool_output_bytes


class RetrievalSession:
    def __init__(self, backend: RetrievalBackend, scope: InvestigationScope,
                 initial_paths: Sequence[str], config: RuntimeConfig, *,
                 seed_ranges: Mapping[str, Sequence[tuple[int, int]]] | None = None,
                 seed_enabled: bool = True):
        self.backend = backend
        self.catalog = backend.catalog
        self.budget = backend.budget
        self.scope = scope
        self.config = config
        self.criteria = {c.criterion_id for c in scope.criteria}
        self.admitted: dict[str, None] = {}
        self.diagnostics: list[dict] = []
        self.retrieval_events: list[dict] = []
        self.known_locations: set[tuple[str, int, int]] = set()
        self.coverage: set[tuple[str, int]] = set()
        self.used_chars = 0
        self.seeds: tuple[EvidenceWindow, ...] = ()
        self.cache = {}
        self.delivered: list[EvidenceWindow] = []
        # State is per run, never shared between modes or tasks. Serial tool
        # dispatch is required because admission and byte budgets are stateful.
        if config.max_parallel_tools != 1 or not config.compact_evidence:
            raise ValueError("v1.6 requires serial tool dispatch and compact evidence")
        if config.max_tool_output_bytes < 4096:
            raise ValueError("v1.6 requires max_tool_output_bytes >= 4096")
        for path in dict.fromkeys(initial_paths):
            if path not in self.catalog.files:
                self.diagnostics.append({"code": "initial_path_outside_catalog"})
                continue
            if len(self.admitted) == self.budget.max_files:
                self.diagnostics.append({"code": "initial_admission_budget_exhausted"})
                break
            self.admitted[path] = None
        self.initial_paths = tuple(self.admitted)
        if seed_enabled:
            self._seed(seed_ranges)

    def _seed(self, ranges):
        seeds = []
        seed_limit = self.budget.max_seed_chars
        for path in self.initial_paths:
            source_lines = self.catalog.files[path].splitlines()
            # A supplied empty range means the baseline did not provide any
            # source range. Do not infer full-file coverage from a path alone.
            requested = (ranges.get(path, ()) if ranges is not None
                         else [(1, min(len(source_lines), self.budget.max_window_lines))])
            for first, last in requested:
                if not 1 <= first <= last <= len(source_lines):
                    self.diagnostics.append({"code": "invalid_initial_range", "path": path})
                    continue
                start = first
                while start <= last and self.used_chars < seed_limit:
                    end = min(last, start + self.budget.max_window_lines - 1)
                    selected = []
                    # Windows contain only complete source lines; no artificial
                    # clipped strings get an evidence ID.
                    for line_number in range(start, end + 1):
                        if (path, line_number) in self.coverage:
                            break
                        line = source_lines[line_number - 1]
                        cost = len(line) + 1
                        if self.used_chars + cost > seed_limit:
                            break
                        selected.append(line)
                        self.coverage.add((path, line_number))
                        self.used_chars += cost
                    if selected:
                        window = EvidenceWindow.from_catalog(self.catalog, f"S{len(seeds)+1:03d}",
                                                             path, start, start + len(selected) - 1)
                        seeds.append(window)
                        start += len(selected)
                    else:
                        # Skip a previously covered or too-large line, but
                        # continue to offer other ranges within the same budget.
                        start += 1
        self.seeds = tuple(seeds)

    def is_covered(self, path, first, last):
        return all((path, i) in self.coverage for i in range(first, last + 1))

    async def retrieve(self, gap: EvidenceGap, *, read_if_unique: bool = False):
        # Only dependency lookup needs a source anchor. Optional irrelevant
        # source hints must not prevent a path or exact lookup.
        if (gap.strategy in {"path", "exact", "hybrid", "symbol"}
                or (gap.strategy == "auto" and not gap.query.startswith(("./", "../", "@/")) and gap.query != "*")) and gap.source_path not in self.admitted:
            gap = gap.model_copy(update={"source_path": None})
        if gap.criterion_id not in self.criteria or (gap.source_path is not None and gap.source_path not in self.admitted):
            status = "unknown_criterion" if gap.criterion_id not in self.criteria else "source_not_admitted"
            self.retrieval_events.append({"criterion_id": gap.criterion_id,
                                          "requested_strategy": gap.strategy, "status": status,
                                          "admitted_paths": [], "candidates": []})
            return {"status": status, "candidates": []}
        key = (gap.strategy, gap.query, gap.source_path)
        if key not in self.cache:
            self.cache[key] = tuple(await self.backend.find(gap))
        hits = self.cache[key]
        # New locators first; known locators are still useful when not yet read.
        hits = sorted(hits, key=lambda item: (
            self.is_covered(item.path, item.line, item.end_line or item.line),
            (item.path, item.line, item.end_line or item.line) in self.known_locations))
        output = {"status": "ok", "criterion_id": gap.criterion_id,
                  "revision": self.catalog.revision, "candidates": []}
        newly_admitted: list[str] = []
        new_locations = 0
        rejected_budget = False
        for hit in hits:
            # Backend is trusted code but still treat its locators as candidates.
            if hit.path not in self.catalog.files:
                continue
            end = hit.end_line or hit.line
            if not 1 <= hit.line <= end <= len(self.catalog.files[hit.path].splitlines()):
                continue
            if hit.path not in self.admitted and len(self.admitted) + len(newly_admitted) >= self.budget.max_files:
                if hit.path not in newly_admitted:
                    rejected_budget = True
                    continue
            loc = (hit.path, hit.line, end)
            covered = self.is_covered(*loc)
            new = loc not in self.known_locations and not covered
            item = {"path": hit.path, "start_line": hit.line, "end_line": end,
                    "channel": hit.channel, "new_locator": new, "already_read": covered}
            tentative = {**output, "candidates": [*output["candidates"], item]}
            # Reserve room for final status/counters and do not mutate admission
            # until the candidate output is known to fit the dispatcher limit.
            worst = {**tentative, "status": "admission_budget_exhausted",
                     "remaining_file_slots": self.budget.max_files}
            if not _fits(worst, "retrieve_evidence", self.config):
                break
            output = tentative
            if hit.path not in self.admitted and hit.path not in newly_admitted:
                newly_admitted.append(hit.path)
            if new:
                new_locations += 1
            if len(output["candidates"]) >= self.budget.max_candidates_per_call:
                break
        for path in newly_admitted:
            self.admitted[path] = None
        for item in output["candidates"]:
            self.known_locations.add((item["path"], item["start_line"], item["end_line"]))
        output["remaining_file_slots"] = self.budget.max_files - len(self.admitted)
        if not new_locations and not newly_admitted:
            output["status"] = ("admission_budget_exhausted" if rejected_budget
                                else "no_new_candidates" if output["candidates"] else "no_matches")
        # Log counts and locators (no code); no oracle input enters this log.
        self.retrieval_events.append({"criterion_id": gap.criterion_id,
                                      "requested_strategy": gap.strategy,
                                      "query": gap.query, "source_path": gap.source_path,
                                      "status": output["status"],
                                      "admitted_paths": newly_admitted,
                                      "candidates": output["candidates"]})
        # Only unique deterministic locators in the authorized retrieval result may be auto-read.
        # Check uncapped locator list, not the admission-limited result list.
        if (read_if_unique and len(hits) == 1 and len(output["candidates"]) == 1
                and hits[0].channel in {"path", "symbol", "dependency"}):
            hit = hits[0]
            saved_coverage, saved_chars, saved_count = set(self.coverage), self.used_chars, len(self.delivered)
            data = self.read(ReadWindowArgs(path=hit.path, start_line=hit.line,
                                            end_line=hit.end_line or hit.line))
            if data.get("success") is not False:
                combined = {**output, "evidence": data}
                if _fits(combined, "retrieve_evidence", self.config):
                    return combined
            # Do not register source that cannot be returned in the tool output.
            self.coverage = saved_coverage
            self.used_chars = saved_chars
            del self.delivered[saved_count:]
        return output

    def read(self, args: ReadWindowArgs):
        path = args.path
        if path not in self.admitted or path not in self.catalog.files:
            self.diagnostics.append({"code": "read_not_admitted"})
            return {"success": False}
        lines = self.catalog.files[path].splitlines()
        start = args.start_line
        requested_end = args.end_line or (start + self.budget.max_window_lines - 1)
        end = min(requested_end, start + self.budget.max_window_lines - 1, len(lines))
        if start > end:
            self.diagnostics.append({"code": "read_range_empty"})
            return {"success": False}
        existing = [window for window in (*self.seeds, *self.delivered)
                    if window.path == path and window.start_line <= end and window.end_line >= start]
        references = [window.evidence_id for window in existing]
        # Return the first uncovered contiguous interval. Later gaps are
        # requested separately using next_line; no already-delivered lines.
        missing = [i for i in range(start, end + 1) if (path, i) not in self.coverage]
        if not missing:
            return {"read_status": "already_covered", "path": path,
                    "start_line": start, "end_line": end,
                    "existing_evidence_ids": references,
                    "next_line": end + 1 if end < len(lines) else None}
        start = missing[0]
        end = next((i - 1 for i in range(start, end + 1) if (path, i) in self.coverage), end)
        evidence_id = f"E{len(self.delivered)+1:03d}"
        selected: list[str] = []
        added_chars = 0
        result = None
        digest = self.catalog.digest(path)
        for number in range(start, end + 1):
            line = lines[number - 1]
            increment = 0 if (path, number) in self.coverage else len(line) + 1
            if self.used_chars + added_chars + increment > self.budget.max_evidence_chars:
                break
            next_lines = [*selected, line]
            data = {"path": path, "revision": self.catalog.revision,
                    "start_line": start, "end_line": number, "lines": next_lines,
                    "numbered_lines": [{"line": i, "content": t}
                                       for i, t in enumerate(next_lines, start)],
                    "sha256": digest, "file_line_count": len(lines),
                    "registered_evidence_id": evidence_id, "existing_evidence_ids": references,
                    "truncated": number < min(requested_end, len(lines)),
                    "next_line": number + 1 if number < len(lines) else None}
            if not _fits(data, "read_file", self.config):
                break
            selected = next_lines
            added_chars += increment
            result = data
        if result is None:
            self.diagnostics.append({"code": "evidence_or_output_budget_exhausted"})
            return {"success": False}
        self.coverage.update((path, i) for i in range(start, result["end_line"] + 1))
        self.used_chars += added_chars
        self.delivered.append(EvidenceWindow.from_catalog(self.catalog, evidence_id, path,
                                                          start, result["end_line"]))
        return result


class RetrieveEvidenceTool(Tool):
    name = "retrieve_evidence"
    description = (
        "Locate evidence by criterion_id and query. With read_if_unique=true, a unique deterministic match also returns committed source evidence. "
        "Use path for exact path, symbol for identifier, dependency for a known import specifier "
        "or '*' (all direct imports, source_path required), exact for literal text, hybrid for semantic search "
        "when enabled, auto for deterministic-first routing. Read only if source evidence was not already returned. Never search out-of-scope questions."
    )
    args_schema = RetrievalRequest

    def __init__(self, session: RetrievalSession):
        self.session = session

    async def execute(self, arguments):
        req = self.args_schema.model_validate(arguments, strict=True)
        criterion = next((c for c in self.session.scope.criteria if c.criterion_id == req.criterion_id), None)
        if criterion is None:
            return {"status": "unknown_criterion", "candidates": []}
        gap = EvidenceGap(criterion_id=req.criterion_id, question=criterion.question,
                          reason=req.reason or "additional code evidence required",
                          strategy=req.strategy, query=req.query, source_path=req.source_path)
        return await self.session.retrieve(gap, read_if_unique=req.read_if_unique)


class ReadEvidenceTool(Tool):
    name = "read_file"
    description = (
        "Read exact numbered committed lines from an admitted candidate. Omit end_line for a bounded window; "
        "maximum 100 lines, actual end_line may be shorter because of budget. Use returned evidence_id for citations. "
        "Do not reread covered lines. A locator alone is not evidence."
    )
    args_schema = ReadWindowArgs

    def __init__(self, session: RetrievalSession):
        self.session = session

    async def execute(self, arguments):
        return self.session.read(self.args_schema.model_validate(arguments, strict=True))


def tool_registry(session: RetrievalSession, *, enabled: bool):
    registry = ToolRegistry()
    if enabled:
        registry.register(RetrieveEvidenceTool(session))
        registry.register(ReadEvidenceTool(session))
    return registry


class ProgressTracker:
    """Runtime hook: only successful new source ranges/locators count as progress."""
    def __init__(self, session: RetrievalSession):
        self.threshold = session.budget.max_no_progress_rounds
        self.coverage = {(w.path, i) for w in session.seeds
                         for i in range(w.start_line, w.end_line + 1)}
        self.locators = set()
        self.cursor = 0
        self.stalls = 0

    def __call__(self, observations: tuple[Observation, ...]) -> str | None:
        if self.cursor == len(observations):
            return None
        new = observations[self.cursor:]
        self.cursor = len(observations)
        progress = False
        for obs in new:
            if not obs.success or not isinstance(obs.data, dict):
                continue
            data = obs.data if obs.tool_name == "read_file" else obs.data.get("evidence", {})
            if isinstance(data, dict) and "lines" in data:
                for line in range(data["start_line"], data["end_line"] + 1):
                    key = (data["path"], line)
                    if key not in self.coverage:
                        self.coverage.add(key)
                        progress = True
            if obs.tool_name == "retrieve_evidence":
                for item in obs.data.get("candidates", []):
                    key = (item["path"], item["start_line"], item["end_line"])
                    if key not in self.locators and item.get("new_locator"):
                        progress = True
                    self.locators.add(key)
        self.stalls = 0 if progress else self.stalls + 1
        return "no_retrieval_progress" if self.stalls >= self.threshold else None
