"""v1.6 offline regression: real Git, real adapters, scripted model responses.

No external model or Benchmark Oracle is needed. Tests are designed for the
FE-Agent checkout after install.py (including its compatibility hooks).
"""
from __future__ import annotations

import asyncio
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agent_core.agent.model import ToolCall
from agent_core.change_surface.model import InvestigationArtifact, ObservedRepositoryFile
from agent_core.agentic_retrieval.catalog import (CatalogError, CommittedCatalog, ReadPolicy,
                                                  assert_clean_revision, safe_path)
from agent_core.agentic_retrieval.contracts import (Criterion, EvidenceGap, InvestigationScope,
                                                    RetrievalBudget)
from agent_core.agentic_retrieval.controller import (AgenticRetrievalController,
                                                      default_runtime_config)
from agent_core.agentic_retrieval.evidence import (CompletionGate, EvidenceError,
                                                    EvidenceLedger, EvidenceWindow)
from agent_core.agentic_retrieval.evaluation import EvaluationOracle, evaluate, summarize
from agent_core.agentic_retrieval.integration import (AgenticInvestigationBuilder,
                                                       maybe_agentic_builder, _source_ranges)
from agent_core.agentic_retrieval.retrieval import Candidate, RetrievalBackend
from agent_core.agentic_retrieval.session import RetrievalSession, ReadWindowArgs, ProgressTracker
from agent_core.llm.model import ChatResponse
from agent_core.runtime.model import Observation, RuntimeConfig
from agent_core.sdd.model import TaskDefinition

API = "src/api/orders.ts"
MOCK = "src/mocks/service.ts"
DB = "src/mocks/database.ts"
OTHER = "src/views/UsersView.vue"
API_TEXT = "import { mockListOrders } from '../mocks/service'\nexport const listOrders = () => mockListOrders()\n"
MOCK_TEXT = "import { orders } from './database'\nexport function mockListOrders() {\n  return orders\n}\n"
DB_TEXT = "export const orders = [{ status: 'paid' }]\n"
REVISION = "a" * 40


def git(root, *args, input=None):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, input=input).stdout.decode().strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "core.autocrlf", "false")
    for path, content in {API: API_TEXT, MOCK: MOCK_TEXT, DB: DB_TEXT,
                          OTHER: "<template>Users</template>\n",
                          "src/.env.local": "secret=do-not-read\n",
                          "src/secrets/credentials.ts": "do-not-read\n",
                          "src2/escape.ts": "do-not-read\n",
                          "benchmark/oracle.json": '{"required_paths":[]}\n'}.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture")
    return root


@pytest.fixture
def catalog(repo):
    return CommittedCatalog.from_git(repo, ReadPolicy(allowed_prefixes=("src",)), RetrievalBudget())


@pytest.fixture
def scope():
    return InvestigationScope(task_id="I-001", objective="Investigate order list",
                              criteria=(Criterion(criterion_id="C01", question="API contract", required_paths=(API,)),
                                        Criterion(criterion_id="C02", question="Mock behavior", required_paths=(MOCK,))),
                              non_goals=("real remote backend implementation",))


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def chat(self, request):
        self.requests.append(request.model_copy(deep=True))
        return next(self.responses)


def call(call_id, name, **arguments):
    return ToolCall(id=call_id, name=name, arguments=arguments)


def response(*calls, content="", tokens=20, **kwargs):
    return ChatResponse(content=content, tool_calls=list(calls), total_tokens=tokens, **kwargs)


def gap(criterion="C02", strategy="dependency", query="../mocks/service", source=API):
    return EvidenceGap(criterion_id=criterion, question="need implementation", reason="missing code evidence",
                       strategy=strategy, query=query, source_path=source)


def retrieve_call(call_id="lookup", **kwargs):
    return call(call_id, "retrieve_evidence", **gap(**kwargs).model_dump(mode="json", exclude_none=True))


def cite(evidence_id="S001", path=API, line=2, quote="listOrders = () => mockListOrders()", **kwargs):
    return dict(evidence_id=evidence_id, path=path, line=line, quote=quote, **kwargs)


def report(*, task_id="I-001", unresolved=False, roles=(), mock_id="E001"):
    c1 = {"criterion_id": "C01", "status": "resolved", "claim": "API calls mock", "citations": [cite()]}
    c2 = {"criterion_id": "C02", "status": "resolved", "claim": "mock returns orders",
          "citations": [cite(mock_id, MOCK, 3, "return orders")]}
    if unresolved:
        c2 = {"criterion_id": "C02", "status": "unresolved", "claim": "need mock implementation",
              "citations": [], "gap": gap().model_dump(mode="json")}
    return {"task_id": task_id, "summary": "investigation", "answers": [c1, c2],
            "observed_files": list(roles), "limitations": ["real backend excluded"]}


def read_observation(catalog, path=MOCK, eid="E001", call_id="read-1", start=1, end=4):
    window = EvidenceWindow.from_catalog(catalog, eid, path, start, end)
    data = window.projection()
    data.pop("evidence_id")
    data.update(revision=catalog.revision, sha256=window.sha256, lines=list(window.lines))
    return Observation(call_id=call_id, tool_name="read_file", success=True, data=data, evidence_id=eid)


def make_gate(catalog, scope, **kwargs):
    seeds = (EvidenceWindow.from_catalog(catalog, "S001", API, 1, 2),)
    return CompletionGate(scope, catalog, seeds, **kwargs)


def run_case(catalog, scope, replies, *, mode="adaptive", budget=None, config=None,
             initial_paths=(API,), **kwargs):
    llm = FakeLLM(replies)
    with catalog.mirror() as mirror:
        backend = RetrievalBackend(catalog, mirror, budget or RetrievalBudget())
        result = asyncio.run(AgenticRetrievalController(llm=llm, runtime_config=config).run(
            backend=backend, scope=scope, initial_paths=initial_paths, mode=mode, **kwargs))
    return result, llm


def code(exc):
    return exc.value.code


# ---- Authorization / immutable source ------------------------------------------------

@pytest.mark.parametrize("path", ["../x.ts", "src/../x.ts", "src\\x.ts", "C:/x.ts", "/src/x.ts", "src//x.ts",
                                   "src/.env.test", "src/.git/config", "src/CON.ts", "src/a. /x.ts",
                                   "src/credentials.ts", "src/oracle.json", "src/x\x00.ts"])
def test_unsafe_path_rejected(path):
    assert not safe_path(path)


def test_policy_requires_authorization():
    with pytest.raises(ValidationError):
        ReadPolicy()
    with pytest.raises(ValidationError):
        ReadPolicy(allowed_paths=("src/.env.local",))


def test_pinned_catalog_ignores_dirty_workspace(repo, catalog):
    (repo / API).write_text("dirty content", encoding="utf-8")
    pinned = CommittedCatalog.from_git(repo, ReadPolicy(allowed_prefixes=("src",)), RetrievalBudget(),
                                       revision=catalog.revision)
    assert pinned.files[API] == API_TEXT
    assert "src2/escape.ts" not in pinned.files
    assert "src/.env.local" not in pinned.files
    assert "src/secrets/credentials.ts" not in pinned.files
    assert "benchmark/oracle.json" not in pinned.files
    with pytest.raises(CatalogError, match="not clean"):
        assert_clean_revision(repo, catalog.revision)
    with pinned.mirror() as mirror:
        assert (mirror / API).read_bytes() == API_TEXT.encode()
        assert not (mirror / ".git").exists()
    assert not mirror.exists()


def test_catalog_revision_is_pinned_across_commits(repo, catalog):
    (repo / API).write_text("new revision", encoding="utf-8")
    git(repo, "add", API)
    git(repo, "commit", "-qm", "next")
    old = CommittedCatalog.from_git(repo, ReadPolicy(allowed_prefixes=("src",)), RetrievalBudget(),
                                    revision=catalog.revision)
    assert old.files[API] == API_TEXT
    with pytest.raises(CatalogError, match="revision mismatch"):
        assert_clean_revision(repo, catalog.revision)


def test_git_symlink_blob_never_admitted(repo):
    # Create a symlink-mode Git blob without requiring Windows symlink privilege.
    oid = git(repo, "hash-object", "-w", "--stdin", input=b"../../secret.txt")
    git(repo, "update-index", "--add", "--cacheinfo", f"120000,{oid},src/link.ts")
    git(repo, "commit", "-qm", "symlink fixture")
    catalog = CommittedCatalog.from_git(repo, ReadPolicy(allowed_prefixes=("src",)), RetrievalBudget())
    assert "src/link.ts" not in catalog.files
    with pytest.raises(CatalogError):
        CommittedCatalog.from_git(repo, ReadPolicy(allowed_paths=("src/link.ts",)), RetrievalBudget())


def test_catalog_budget_fail_closed(repo):
    with pytest.raises(CatalogError, match="catalog exceeds"):
        CommittedCatalog.from_git(repo, ReadPolicy(allowed_prefixes=("src",)), RetrievalBudget(max_catalog_files=1))


def test_explicit_missing_file_fail_closed(repo):
    with pytest.raises(CatalogError):
        CommittedCatalog.from_git(repo, ReadPolicy(allowed_paths=(API, "src/missing.ts")), RetrievalBudget())


# ---- Scope and evidence gate ----------------------------------------------------------


def test_scope_from_existing_task():
    task = TaskDefinition(task_id="INV-001", task_type="investigation", title="orders", objective="inspect orders",
                          source_step_order=1, investigation_questions=["API?", "Mock?", "API?"])
    actual = InvestigationScope.from_task(task, SimpleNamespace(non_goals=["backend"], constraints=["read only"]))
    assert [item.question for item in actual.criteria] == ["API?", "Mock?"]
    assert actual.non_goals == ("backend",)
    with pytest.raises(ValidationError):
        InvestigationScope(task_id="I", objective="o", criteria=(Criterion(criterion_id="C", question="q"),)*2)


def test_valid_report_and_markdown_normalization(catalog, scope):
    gate = make_gate(catalog, scope)
    payload = "```json\n" + json.dumps(report()) + "\n```"
    assert len(gate.validate(payload, [read_observation(catalog)]).answers) == 2


@pytest.mark.parametrize("kind,expected", [
    ("missing", "criterion_coverage_mismatch"), ("duplicate", "criterion_coverage_mismatch"),
    ("extra", "criterion_coverage_mismatch"), ("wrong_task", "report_identity_mismatch"),
    ("bad_id", "invalid_citation"), ("wrong_line", "citation_text_mismatch"),
    ("bad_quote", "citation_text_mismatch"), ("required_path", "required_evidence_missing"),
    ("unresolved", "unresolved_criteria"), ("extra_text", "invalid_report_schema"),
])
def test_gate_rejects_bad_reports(catalog, scope, kind, expected):
    value = report()
    if kind == "missing": value["answers"].pop()
    if kind == "duplicate": value["answers"][1] = copy.deepcopy(value["answers"][0])
    if kind == "extra": value["answers"][1]["criterion_id"] = "C99"
    if kind == "wrong_task": value["task_id"] = "WRONG"
    if kind == "bad_id": value["answers"][1]["citations"][0]["evidence_id"] = "E999"
    if kind == "wrong_line": value["answers"][1]["citations"][0]["line"] = 2
    if kind == "bad_quote": value["answers"][1]["citations"][0]["quote"] = "invented"
    if kind == "required_path": value["answers"][1]["citations"] = [cite()]
    if kind == "unresolved": value = report(unresolved=True)
    encoded = json.dumps(value)
    if kind == "extra_text": encoded = "result:\n" + encoded
    with pytest.raises(EvidenceError) as exc:
        make_gate(catalog, scope).validate(encoded, [read_observation(catalog)])
    assert code(exc) == expected


def test_forged_snapshot_or_duplicate_observation_rejected(catalog, scope):
    original = read_observation(catalog)
    fake = original.model_copy(deep=True)
    fake.data["revision"] = "b" * 40
    with pytest.raises(EvidenceError) as exc:
        make_gate(catalog, scope).validate(json.dumps(report()), [fake])
    assert code(exc) == "snapshot_evidence_mismatch"
    with pytest.raises(EvidenceError) as exc:
        make_gate(catalog, scope).validate(json.dumps(report()), [original, original])
    assert code(exc) == "duplicate_call_id"


def test_failed_read_cannot_become_evidence(catalog, scope):
    failed = read_observation(catalog).model_copy(update={"success": False})
    with pytest.raises(EvidenceError) as exc:
        make_gate(catalog, scope).validate(json.dumps(report()), [failed])
    assert code(exc) == "invalid_citation"


def test_multiline_quote_requires_actual_range(catalog, scope):
    value = report()
    value["answers"][1]["citations"] = [cite("E001", MOCK, 2,
        "export function mockListOrders() {\n  return orders", end_line=3)]
    gate = make_gate(catalog, scope)
    gate.validate(json.dumps(value), [read_observation(catalog)])
    value["answers"][1]["citations"][0]["quote"] = "return orders"
    with pytest.raises(EvidenceError) as exc:
        gate.validate(json.dumps(value), [read_observation(catalog)])
    assert code(exc) == "citation_text_mismatch"


def roles():
    return [dict(relative_path=API, observed_role="candidate_target", rationale="API contract needs change",
                 confidence=.9, criterion_ids=["C01"], citations=[cite()]),
            dict(relative_path=MOCK, observed_role="candidate_target", rationale="mock behavior needs change",
                 confidence=.9, criterion_ids=["C02"], citations=[cite("E001", MOCK, 3, "return orders")])]


def test_new_target_requires_separate_approval(catalog, scope):
    content = json.dumps(report(roles=roles()))
    obs = [read_observation(catalog)]
    make_gate(catalog, scope, require_roles=True).validate(content, obs)
    with pytest.raises(EvidenceError) as exc:
        make_gate(catalog, scope, require_roles=True, candidate_target_ceiling=frozenset({API})).validate(content, obs)
    assert code(exc) == "new_target_requires_approval"
    make_gate(catalog, scope, require_roles=True, candidate_target_ceiling=frozenset({API, MOCK})).validate(content, obs)


def test_role_requires_same_file_and_coverage(catalog, scope):
    obs = [read_observation(catalog)]
    value = report(roles=roles())
    value["observed_files"][1]["citations"] = [cite()]
    with pytest.raises(EvidenceError) as exc:
        make_gate(catalog, scope, require_roles=True).validate(json.dumps(value), obs)
    assert code(exc) == "invalid_role_evidence"
    value = report(roles=roles()[:1])
    with pytest.raises(EvidenceError) as exc:
        make_gate(catalog, scope, require_roles=True).validate(json.dumps(value), obs)
    assert code(exc) == "missing_file_assessment"


# ---- Existing retrieval adapters and actual Runtime loop ------------------------------


def test_adaptive_fast_path_uses_seed_only(catalog, scope):
    value = report(mock_id="S002")
    result, llm = run_case(catalog, scope, [response(content=json.dumps(value))], initial_paths=(API, MOCK))
    assert result.runtime.status == "completed"
    assert result.runtime.tool_calls == 0 and result.runtime.steps == 1
    assert result.report is not None
    assert len(llm.requests) == 1


def test_dynamic_graph_expansion(catalog, scope):
    result, llm = run_case(catalog, scope, [response(retrieve_call()),
                                          response(call("read", "read_file", path=MOCK)),
                                          response(content=json.dumps(report()))])
    assert result.runtime.status == "completed", result.runtime.model_dump()
    assert set(result.evidence_paths) == {API, MOCK}
    assert result.initial_paths == (API,)
    assert result.runtime.tool_calls == 2
    assert result.retrieval_events[0]["candidates"][0]["channel"] == "dependency"
    assert result.runtime.observations[1].evidence_id == "E001"
    outputs = [item for item in llm.requests[2].input_items if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in outputs] == ["lookup", "read"]
    read_payload = json.loads(outputs[1]["output"])
    assert read_payload["evidence_id"] == "E001" and "lines" not in read_payload["data"]


def test_multihop_gap_expansion(catalog, scope):
    extra = Criterion(criterion_id="C03", question="mock data source", required_paths=(DB,))
    scope = scope.model_copy(update={"criteria": (*scope.criteria, extra)})
    value = report()
    value["answers"].append(dict(criterion_id="C03", status="resolved", claim="database holds paid order",
                                 citations=[cite("E002", DB, 1, "status: 'paid'")]))
    result, _ = run_case(catalog, scope, [response(retrieve_call()), response(call("r1", "read_file", path=MOCK)),
        response(retrieve_call("lookup2", criterion="C03", query="./database", source=MOCK)),
        response(call("r2", "read_file", path=DB)), response(content=json.dumps(value))])
    assert result.runtime.status == "completed", result.runtime.model_dump()
    assert set(result.evidence_paths) == {API, MOCK, DB}
    assert result.runtime.tool_calls == 4


def test_fixed_context_blocks_missing_required_evidence(catalog, scope):
    result, llm = run_case(catalog, scope, [response(content=json.dumps(report(unresolved=True)))], mode="fixed_context")
    assert result.runtime.status == "blocked"
    assert result.validation_code == "unresolved_criteria"
    assert result.runtime.tool_calls == 0 and llm.requests[0].tools is None
    assert result.runtime.candidate_output and result.report is None


def test_unauthorized_path_never_becomes_evidence(catalog, scope):
    result, _ = run_case(catalog, scope, [response(retrieve_call(strategy="path", query="src/.env.local", source=None)),
        response(call("r", "read_file", path="src/.env.local")), response(content=json.dumps(report(unresolved=True)))])
    assert result.runtime.status == "blocked"
    assert "src/.env.local" not in result.admitted_paths
    assert all(o.tool_name != "read_file" or not o.success for o in result.runtime.observations)


def test_read_without_admission_rejected(catalog, scope):
    result, _ = run_case(catalog, scope, [response(call("r", "read_file", path=MOCK)),
        response(content=json.dumps(report(unresolved=True)))])
    assert not result.runtime.observations[0].success
    assert result.runtime.status == "blocked"


def test_file_admission_budget_atomic(catalog, scope):
    budget = RetrievalBudget(max_files=1, initial_max_files=1)
    result, _ = run_case(catalog, scope, [response(retrieve_call()), response(content=json.dumps(report(unresolved=True)))], budget=budget)
    assert result.admitted_paths == (API,)
    assert result.retrieval_events[0]["status"] == "admission_budget_exhausted"
    assert result.runtime.status == "blocked"


def test_no_progress_triggers_finalization(catalog, scope):
    result, llm = run_case(catalog, scope, [
        response(retrieve_call("x1", strategy="exact", query="__missing_one__", source=None)),
        response(retrieve_call("x2", strategy="exact", query="__missing_two__", source=None)),
        response(content=json.dumps(report(unresolved=True)))])
    assert result.runtime.status == "blocked" and result.validation_code == "unresolved_criteria"
    assert llm.requests[-1].tool_choice == "none"
    assert any(e.event == "finalization_started" and e.data["reason"] == "no_retrieval_progress"
               for e in result.runtime.trace)


def test_provider_cannot_execute_tools_during_early_finalization(catalog, scope):
    budget = RetrievalBudget(max_no_progress_rounds=1)
    result, _ = run_case(catalog, scope, [response(retrieve_call(strategy="path", query="src/missing.ts", source=None)),
        response(call("bad", "read_file", path=API))], budget=budget)
    assert result.runtime.reason == "tool_calls_during_finalization"
    assert result.runtime.tool_calls == 1 and len(result.runtime.observations) == 1


def test_unknown_criterion_cannot_expand_scope(catalog, scope):
    result, _ = run_case(catalog, scope, [response(retrieve_call(criterion="C99", strategy="path", query=MOCK, source=None)),
        response(content=json.dumps(report(unresolved=True)))])
    assert result.admitted_paths == (API,)
    assert result.runtime.observations[0].data["status"] == "unknown_criterion"


def test_evidence_and_tool_byte_budget(catalog, scope):
    budget = RetrievalBudget(max_seed_chars=0, max_evidence_chars=65)
    config = default_runtime_config().model_copy(update={"max_tool_output_bytes":4096})
    with catalog.mirror() as root:
        session = RetrievalSession(RetrievalBackend(catalog, root, budget), scope, (API,), config)
        data = session.read(ReadWindowArgs(path=API))
        assert data["end_line"] == 1 and data["truncated"]
        before = session.used_chars
        session.read(ReadWindowArgs(path=API, start_line=1, end_line=1))
        assert session.used_chars == before
        assert session.read(ReadWindowArgs(path=API, start_line=2))["success"] is False


def test_full_agentic_no_seed_text(catalog, scope):
    value = report()
    value["answers"][0]["citations"] = [cite("E001")]
    value["answers"][1]["citations"] = [cite("E002", MOCK, 3, "return orders")]
    result, llm = run_case(catalog, scope, [response(call("r1", "read_file", path=API)),
        response(retrieve_call()), response(call("r2", "read_file", path=MOCK)),
        response(content=json.dumps(value))], mode="full_agentic")
    assert result.runtime.status == "completed"
    assert result.seed_evidence == ()
    assert json.loads(llm.requests[0].messages[1].content)["seed_evidence"] == []


def test_runtime_token_budget_not_bypassed(catalog, scope):
    config = default_runtime_config().model_copy(update={"max_total_tokens": 100})
    result, _ = run_case(catalog, scope, [response(retrieve_call(), tokens=101)], config=config)
    assert result.runtime.reason == "token_budget_exhausted" and not result.runtime.observations


def test_existing_exact_and_symbol_routes(catalog, scope):
    with catalog.mirror() as root:
        backend = RetrievalBackend(catalog, root, RetrievalBudget())
        exact = asyncio.run(backend.find(gap(strategy="exact", query="return orders", source=None)))
        assert exact[0].path == MOCK and exact[0].line == 3
        symbol = asyncio.run(backend.find(gap(strategy="symbol", query="mockListOrders")))
        assert any(item.path == MOCK for item in symbol)
        # When the actual parser is installed, prefer a definition locator.
        if "symbol_snapshot_unavailable" not in backend.diagnostics:
            assert any(item.path == MOCK and item.channel == "symbol" for item in symbol)


def test_hybrid_reuses_semantic_index(catalog, scope):
    class Embedding:
        async def embed(self, texts):
            return [[1.0, 0.0] if "path: " + MOCK in t or t == "semantic need" else [0.0, 1.0] for t in texts]
    async def scenario(root):
        backend = RetrievalBackend(catalog, root, RetrievalBudget())
        await backend.prepare_semantic(Embedding())
        hits = await backend.find(gap(strategy="hybrid", query="semantic need", source=None))
        assert hits and hits[0].path == MOCK and hits[0].channel == "hybrid"
    with catalog.mirror() as root:
        asyncio.run(scenario(root))


def test_initial_retrieval_reuses_task_service(catalog):
    task = TaskDefinition(task_id="I-001", task_type="investigation", title="listOrders",
                          objective="listOrders", source_step_order=1, investigation_questions=["listOrders"])
    with catalog.mirror() as root:
        backend = RetrievalBackend(catalog, root, RetrievalBudget())
        paths = asyncio.run(backend.initial_candidates(task=task, raw_requirement="listOrders"))
    assert API in paths
    assert len(paths) <= RetrievalBudget().initial_max_files


# ---- Harness wrapper and offline benchmark -------------------------------------------


class DummyBundle:
    def __init__(self):
        self.task = SimpleNamespace(task_id="I-001", task_type="investigation", title="orders", objective="orders",
                                    investigation_questions=["API contract", "Mock behavior"], responsibilities=[])
        self.requirement = SimpleNamespace(non_goals=["real backend"], constraints=[])
        self.code_context = [SimpleNamespace(relative_path=API, content_mode="full_file", content=API_TEXT,
                                             start_line=1, end_line=2, segments=[])]

    def model_copy(self, *, deep):
        return copy.deepcopy(self)


class BaselineBuilder:
    def __init__(self):
        self.llm = FakeLLM([])
        self.calls = 0
        self.artifact = InvestigationArtifact(task_id="I-001", summary="baseline",
            observed_files=[ObservedRepositoryFile(relative_path=API, observed_role="candidate_target", evidence=["baseline evidence"], confidence=.9)])

    async def build(self, *, bundle):
        self.calls += 1
        return self.artifact

    @staticmethod
    def _build_payload(*, bundle):
        return {"repository": {"root_path": "private-root"}, "code_context": [{"content": "legacy"}],
                "upstream_artifacts": [], "context_budget": {}}

    @staticmethod
    def _build_system_prompt(*, schema):
        return "role policy\nJSON Schema：\n{}"


def test_shadow_failure_returns_exact_baseline(repo, catalog, tmp_path, monkeypatch):
    base = BaselineBuilder()
    bundle = DummyBundle()
    log = tmp_path / "shadow.jsonl"
    builder = AgenticInvestigationBuilder(baseline_builder=base, repository_root=repo,
        expected_revision=catalog.revision, read_policy=ReadPolicy(allowed_prefixes=("src",)),
        mode="shadow", report_path=log)
    async def fail(_):
        raise RuntimeError("diagnostic")
    monkeypatch.setattr(builder, "_candidate", fail)
    artifact = asyncio.run(builder.build(bundle=bundle))
    assert artifact is base.artifact and base.calls == 1
    assert bundle.code_context[0].content == API_TEXT
    assert json.loads(log.read_text(encoding="utf-8"))["status"] == "failed"


def test_active_does_not_fallback_on_failure(repo, catalog, tmp_path, monkeypatch):
    base = BaselineBuilder()
    builder = AgenticInvestigationBuilder(baseline_builder=base, repository_root=repo,
        expected_revision=catalog.revision, read_policy=ReadPolicy(allowed_prefixes=("src",)),
        mode="active", report_path=tmp_path / "active.jsonl")
    async def fail(_):
        raise RuntimeError("blocked")
    monkeypatch.setattr(builder, "_candidate", fail)
    with pytest.raises(RuntimeError, match="blocked"):
        asyncio.run(builder.build(bundle=DummyBundle()))
    assert base.calls == 0


def test_active_integration_with_verified_new_dependency(repo, catalog, tmp_path):
    base = BaselineBuilder()
    value = report(roles=roles())
    value["observed_files"][1]["observed_role"] = "dependency"
    runtime_llm = FakeLLM([response(retrieve_call()), response(call("r", "read_file", path=MOCK)),
                           response(content=json.dumps(value))])
    builder = AgenticInvestigationBuilder(baseline_builder=base, repository_root=repo,
        expected_revision=catalog.revision, read_policy=ReadPolicy(allowed_prefixes=("src",)),
        mode="active", report_path=tmp_path / "active.jsonl", runtime_llm=runtime_llm)
    artifact = asyncio.run(builder.build(bundle=DummyBundle()))
    assert base.calls == 0
    assert artifact.task_id == "I-001"
    assert [i.relative_path for i in artifact.observed_files] == [API, MOCK]
    assert artifact.observed_files[1].observed_role == "dependency"
    assert artifact.read_only_paths == [MOCK]
    assert "return orders" in artifact.observed_files[1].evidence[0]
    assert json.loads((tmp_path / "active.jsonl").read_text(encoding="utf-8"))["status"] == "completed"


def test_source_range_mismatch_blocks_integration(catalog):
    bundle = DummyBundle()
    bundle.code_context[0].content = "wrong commit"
    with pytest.raises(EvidenceError) as exc:
        _source_ranges(bundle, catalog)
    assert code(exc) == "baseline_source_mismatch"


def test_default_off_builder_is_unchanged(monkeypatch):
    monkeypatch.delenv("FE_AGENT_V16_MODE", raising=False)
    base = BaselineBuilder()
    assert maybe_agentic_builder(baseline_builder=base, repository_root=None,
                                 expected_revision=None, report_path=None) is base


def test_oracle_only_used_after_runtime(catalog, scope):
    result, llm = run_case(catalog, scope, [response(content=json.dumps(report(mock_id="S002")))], initial_paths=(API, MOCK))
    # Deliberately different offline expectation to demonstrate false admission
    # evaluation, not a requirement that was sent to the model.
    oracle = EvaluationOracle(required_paths=(API, MOCK, DB))
    metrics = evaluate(result, oracle)
    assert metrics["false_admission"] is True
    assert metrics["required_file_recall"] == pytest.approx(2/3)
    payload = json.loads(llm.requests[0].messages[1].content)
    assert "oracle" not in payload and DB not in payload["initial_paths"]
    assert summarize([metrics])["adaptive"]["runs"] == 1


def test_new_test_role_requires_approval(catalog, scope):
    value = report(roles=roles())
    value['observed_files'][1]['observed_role'] = 'test'
    with pytest.raises(EvidenceError):
        make_gate(catalog, scope, require_roles=True,
                  candidate_target_ceiling=frozenset({API})).validate(
            json.dumps(value), [read_observation(catalog)])


def test_read_only_provenance_survives_serialization_and_surface_guard():
    from agent_core.agentic_retrieval.write_guard import assert_read_only_paths
    from agent_core.change_surface.errors import ChangeSurfaceResolutionError
    observed = ObservedRepositoryFile(relative_path=MOCK, observed_role='dependency',
                                      evidence=['verified source'], confidence=0.9)
    legacy = InvestigationArtifact(task_id='I', summary='legacy', observed_files=[observed])
    assert 'read_only_paths' not in legacy.model_dump()
    assert 'read_only_paths' not in InvestigationArtifact.model_json_schema()['properties']
    artifact = InvestigationArtifact(task_id='I', summary='verified', observed_files=[observed],
                                     read_only_paths=[MOCK])
    restored = InvestigationArtifact.model_validate_json(artifact.model_dump_json())
    assert restored.read_only_paths == [MOCK]
    for field in ('implementation_targets', 'tests', 'create_files'):
        surface = SimpleNamespace(implementation_targets=[], tests=[], create_files=[])
        setattr(surface, field, ['./' + MOCK.upper()])
        with pytest.raises(ChangeSurfaceResolutionError):
            assert_read_only_paths(surface, [restored])
    assert_read_only_paths(SimpleNamespace(implementation_targets=[API], tests=[], create_files=[]), [restored])
    with pytest.raises(ValidationError):
        InvestigationArtifact(task_id='I', summary='invalid', read_only_paths=[MOCK],
                              observed_files=[observed.model_copy(update={'observed_role':'candidate_target'})])
