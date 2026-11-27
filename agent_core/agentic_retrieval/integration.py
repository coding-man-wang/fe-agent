"""Opt-in adapter for HarnessV1Coordinator.investigation_builder.

Shadow returns exactly the original builder's result; active consumes only a
validated report. Neither mode changes the patch writer or Scope Guard.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Literal

from agent_core.change_surface.errors import InvestigationArtifactBuildError
from agent_core.change_surface.investigation import InvestigationArtifactBuilder
from agent_core.change_surface.model import InvestigationArtifact, ObservedRepositoryFile
from agent_core.llm.retrying import RetryingLLM
from agent_core.runtime.model import RuntimeConfig

from .catalog import CommittedCatalog, ReadPolicy, assert_clean_revision
from .contracts import InvestigationReport, InvestigationScope, RetrievalBudget
from .controller import AgenticRetrievalController
from .evidence import EvidenceError
from .retrieval import RetrievalBackend

logger = logging.getLogger(__name__)


def _source_ranges(bundle, catalog: CommittedCatalog):
    """Use baseline range metadata; re-read only complete committed lines.

    Full files are checked against their original text (newline normalization
    is allowed). Partial/symbol windows are rebuilt from declared source ranges,
    not from rendered context text or a guessed line mapping.
    """
    paths = []
    ranges: dict[str, list[tuple[int, int]]] = {}
    for item in bundle.code_context:
        path = item.relative_path
        if path not in paths:
            paths.append(path)
        if path not in catalog.files:
            continue
        lines = catalog.files[path].splitlines()
        ranges.setdefault(path, [])
        if item.content_mode == "full_file":
            if item.content.splitlines() != lines:
                raise EvidenceError("baseline_source_mismatch", "baseline file differs from pinned commit")
            if lines:
                ranges[path].append((1, len(lines)))
        elif item.content_mode == "symbol_window":
            for segment in item.segments:
                if not 1 <= segment.start_line <= segment.end_line <= len(lines):
                    raise EvidenceError("baseline_range_mismatch", "invalid source segment")
                ranges[path].append((segment.start_line, segment.end_line))
        else:
            # Existing char-clipped context can end inside a line. Only accept
            # complete baseline lines as seed evidence; clipped tail is omitted.
            if item.start_line is None or item.end_line is None:
                continue
            start, end = item.start_line, item.end_line
            if not 1 <= start <= end <= len(lines):
                raise EvidenceError("baseline_range_mismatch", "invalid source range")
            delivered = item.content.splitlines()
            source = lines[start - 1:end]
            complete = 0
            for original, actual in zip(source, delivered):
                if original != actual:
                    break
                complete += 1
            if complete:
                ranges[path].append((start, start + complete - 1))
            # A mismatch before the last delivered line is not normal clipping.
            if complete < max(0, len(delivered) - 1) or len(delivered) > len(source):
                raise EvidenceError("baseline_source_mismatch", "baseline range differs from pinned source")
    return tuple(paths), ranges


def _role_policy(builder: InvestigationArtifactBuilder) -> str:
    # Preserve the actual existing Responsibility Ownership / cross-layer /
    # minimal-change policy. The v1.6 prompt defines the new JSON contract.
    prompt = builder._build_system_prompt(schema={})
    marker = "\nJSON Schema："
    if marker not in prompt:
        raise ValueError("InvestigationArtifactBuilder prompt contract changed")
    return prompt.rsplit(marker, 1)[0].strip()


def _safe_model_context(builder, bundle):
    payload = builder._build_payload(bundle=bundle)
    payload.pop("code_context", None)
    payload.pop("context_budget", None)
    if isinstance(payload.get("repository"), dict):
        payload["repository"].pop("root_path", None)
    # Upstream artifacts are application metadata, not verified current-source
    # evidence. Preserve execution identity/paths only, not embedded code or
    # previous model claims that may fall outside this read policy.
    upstream = []
    for item in payload.get("upstream_artifacts", []):
        if not isinstance(item, dict):
            continue
        metadata = {k: item[k] for k in ("artifact_type", "task_id", "repository_revision")
                    if k in item and isinstance(item[k], str)}
        data = item.get("payload", item.get("data", item))
        if isinstance(data, dict):
            for k in ("task_id", "repository_revision"):
                if k in data and isinstance(data[k], str):
                    metadata[k] = data[k]
            if metadata.get("artifact_type") == "execution" and isinstance(data.get("changed_files"), list):
                metadata["changed_files"] = [
                    p if isinstance(p, str) else {"relative_path": p["relative_path"]}
                    for p in data["changed_files"]
                    if isinstance(p, str) or (isinstance(p, dict) and isinstance(p.get("relative_path"), str))
                ]
        if metadata:
            upstream.append(metadata)
    payload["upstream_artifacts"] = upstream
    return payload


def _cite(citation) -> str:
    end = citation.end_line or citation.line
    location = str(citation.line) if end == citation.line else f"{citation.line}-{end}"
    return f"[{citation.evidence_id}] {citation.path}:{location}\n{citation.quote}"


def to_artifact(report: InvestigationReport, *, read_only_paths=()) -> InvestigationArtifact:
    """Caller must first pass CompletionGate; do not map locators to write roles."""
    if any(answer.status != "resolved" for answer in report.answers):
        raise ValueError("unresolved report cannot be consumed")
    return InvestigationArtifact(
        task_id=report.task_id,
        summary="；".join(answer.claim for answer in report.answers),
        observed_files=[ObservedRepositoryFile(
            relative_path=item.relative_path, observed_role=item.observed_role,
            confidence=item.confidence,
            evidence=[f"{item.rationale}\n{_cite(citation)}" for citation in item.citations])
            for item in report.observed_files],
        findings=[f"{answer.claim}\n" + "\n".join(_cite(c) for c in answer.citations)
                  for answer in report.answers],
        unresolved_questions=[],
        read_only_paths=list(read_only_paths),
    )


def _targets(artifact):
    return sorted(item.relative_path for item in artifact.observed_files
                  if item.observed_role == "candidate_target") if artifact is not None else []


class AgenticInvestigationBuilder:
    def __init__(self, *, baseline_builder: InvestigationArtifactBuilder,
                 repository_root: str | Path, expected_revision: str, read_policy: ReadPolicy,
                 mode: Literal["shadow", "active"], report_path: Path,
                 budget: RetrievalBudget | None = None, runtime_config: RuntimeConfig | None = None,
                 runtime_llm=None, embedding_provider=None,
                 approved_target_candidates: tuple[str, ...] = (), aliases=None):
        if mode not in {"shadow", "active"}:
            raise ValueError("invalid integration mode")
        if any(not read_policy.permits(path) for path in approved_target_candidates):
            raise ValueError("approved target candidate is outside read policy")
        self.baseline = baseline_builder
        self.llm = baseline_builder.llm  # preserve existing builder interface
        self.runtime_llm = runtime_llm if runtime_llm is not None else self.llm
        # Runtime owns retries. The ordinary baseline builder keeps its own policy.
        while isinstance(self.runtime_llm, RetryingLLM):
            self.runtime_llm = self.runtime_llm.llm
        self.root = Path(repository_root).resolve(strict=True)
        self.revision = expected_revision
        self.policy = read_policy
        self.mode = mode
        self.report_path = Path(report_path).resolve()
        if self.report_path.is_relative_to(self.root):
            raise ValueError("audit path must be outside the investigated repository")
        self.budget = budget or RetrievalBudget()
        self.config = runtime_config
        self.embedding_provider = embedding_provider
        self.approved_target_candidates = frozenset(approved_target_candidates)
        self.aliases = aliases

    def _audit(self, data: dict):
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")

    async def _candidate(self, bundle):
        # The legacy fresh preparation uses the working tree for Baseline.
        # Require it to match the approved commit before treating its selection
        # as a comparable initial context. Candidate source itself is pinned.
        assert_clean_revision(self.root, self.revision)
        catalog = CommittedCatalog.from_git(self.root, self.policy, self.budget,
                                            revision=self.revision)
        scope = InvestigationScope.from_task(bundle.task, bundle.requirement)
        initial_paths, seed_ranges = _source_ranges(bundle, catalog)
        if self.mode == "active":
            # Reading a new file does not automatically extend the original
            # investigation's candidate-target upper bound. Newly discovered
            # write candidates require a separate host approval.
            ceiling = frozenset(path for path in initial_paths if path in catalog.files) | self.approved_target_candidates
        else:
            ceiling = None
        with catalog.mirror() as root:
            backend = RetrievalBackend(catalog, root, self.budget, aliases=self.aliases)
            await backend.prepare_semantic(self.embedding_provider)
            result = await AgenticRetrievalController(llm=self.runtime_llm, runtime_config=self.config).run(
                backend=backend, scope=scope, initial_paths=initial_paths, seed_ranges=seed_ranges,
                mode="adaptive", model_context=_safe_model_context(self.baseline, bundle),
                role_policy=_role_policy(self.baseline), require_roles=True,
                candidate_target_ceiling=ceiling)
        assert_clean_revision(self.root, self.revision)
        candidate = None
        if result.report is not None and result.runtime.status == "completed":
            read_only = sorted(item.relative_path for item in result.report.observed_files
                               if ceiling is not None and item.relative_path not in ceiling)
            candidate = to_artifact(result.report, read_only_paths=read_only)
        return result, candidate

    async def build(self, *, bundle) -> InvestigationArtifact:
        # Shadow uses the original result object and never puts the candidate
        # artifact into the run-scoped ArtifactStore.
        baseline = await self.baseline.build(bundle=bundle) if self.mode == "shadow" else None
        started = perf_counter()
        data = {"task_id": bundle.task.task_id, "revision": self.revision, "mode": self.mode,
                "baseline_targets": _targets(baseline)}
        candidate = None
        try:
            result, candidate = await self._candidate(bundle.model_copy(deep=True))
            data.update(status=result.runtime.status, reason=result.runtime.reason,
                        validation_code=result.validation_code, reported_tokens=result.runtime.reported_tokens,
                        usage_complete=result.runtime.usage_complete, steps=result.runtime.steps,
                        tool_calls=result.runtime.tool_calls, initial_paths=list(result.initial_paths),
                        admitted_paths=list(result.admitted_paths), evidence_paths=list(result.evidence_paths),
                        candidate_targets=_targets(candidate),
                        read_only_paths=list(candidate.read_only_paths) if candidate is not None else [],
                        added_targets=sorted(set(_targets(candidate)) - set(_targets(baseline))) if baseline is not None else [],
                        removed_targets=sorted(set(_targets(baseline)) - set(_targets(candidate))) if baseline is not None and candidate is not None else [],
                        retrieval_events=result.retrieval_events, diagnostics=result.diagnostics)
            if candidate is None and self.mode == "active":
                raise InvestigationArtifactBuildError(f"v1.6 investigation blocked: {result.validation_code or result.runtime.reason}")
        except Exception as exc:
            if "status" not in data:
                data.update(status="failed", reason=getattr(exc, "code", type(exc).__name__))
            data["error_type"] = type(exc).__name__
            if self.mode == "active":
                data["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
                self._audit(data)
                raise
        data["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
        if self.mode == "shadow":
            try:
                self._audit(data)
            except Exception:
                logger.warning("v1.6 shadow audit failed", exc_info=True)
            assert baseline is not None
            return baseline
        self._audit(data)
        if candidate is None:
            raise InvestigationArtifactBuildError("v1.6 investigation has no accepted artifact")
        # Recheck just before crossing the existing Harness artifact boundary.
        assert_clean_revision(self.root, self.revision)
        return candidate


def make_embedding_provider(enabled: bool):
    if not enabled:
        return None
    from agent_core.codebase.openai_embedding import OpenAIEmbeddingProvider
    return OpenAIEmbeddingProvider(model=os.environ["EMBEDDING_MODEL_ID"],
                                   api_key=os.environ["EMBEDDING_API_KEY"],
                                   base_url=os.getenv("EMBEDDING_BASE_URL"), batch_size=20)


def maybe_agentic_builder(*, baseline_builder, repository_root, expected_revision, report_path):
    """Default off; explicit host-only environment settings for fresh Harness runs."""
    mode = os.getenv("FE_AGENT_V16_MODE", "off").strip().lower()
    if mode == "off":
        return baseline_builder
    if mode not in {"shadow", "active"}:
        raise ValueError("FE_AGENT_V16_MODE must be off, shadow or active")

    def values(key):
        return tuple(x.strip().rstrip("/") for x in os.getenv(key, "").split(";") if x.strip())

    return AgenticInvestigationBuilder(
        baseline_builder=baseline_builder, repository_root=repository_root,
        expected_revision=expected_revision,
        read_policy=ReadPolicy(allowed_prefixes=values("FE_AGENT_V16_READ_PREFIXES"),
                               allowed_paths=values("FE_AGENT_V16_READ_FILES")),
        mode=mode, report_path=report_path,
        approved_target_candidates=values("FE_AGENT_V16_APPROVED_TARGET_CANDIDATES"),
        embedding_provider=make_embedding_provider(os.getenv("FE_AGENT_V16_ENABLE_SEMANTIC") == "1"))
