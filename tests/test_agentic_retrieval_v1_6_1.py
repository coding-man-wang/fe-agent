"""Regression tests for the v1.6.1 efficiency changes; no API key needed."""
from __future__ import annotations

import asyncio
import json
import subprocess
from types import SimpleNamespace

import pytest

from agent_core.agent.model import ToolCall
from agent_core.agentic_retrieval.budget_runtime import estimated_input_tokens
from agent_core.agentic_retrieval.catalog import CommittedCatalog, ReadPolicy
from agent_core.agentic_retrieval.contracts import Criterion, EvidenceGap, InvestigationScope, RetrievalBudget, RetrievalRequest
from agent_core.agentic_retrieval.controller import AgenticRetrievalController, default_runtime_config
from agent_core.agentic_retrieval.evidence import EvidenceError, EvidenceLedger
from agent_core.agentic_retrieval.retrieval import Candidate, RetrievalBackend, authorized_path_matches, identifiers_in
from agent_core.agentic_retrieval.session import RetrievalSession, ReadWindowArgs
from agent_core.llm.model import ChatResponse
from agent_core.runtime.model import Observation

API = "src/api/orders.ts"
MOCK = "src/mocks/service.ts"
SRC = "import { mockListOrders } from '../mocks/service'\nexport const listOrders = () => mockListOrders()\n"
MOCK_SRC = "export function mockListOrders() {\n  return []\n}\n"


@pytest.fixture
def catalog(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "core.autocrlf", "false")
    for path, source in {API: SRC, MOCK: MOCK_SRC, "src/.env.local": "secret=test\n"}.items():
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(source, encoding="utf-8", newline="\n")
    git("add", ".")
    git("commit", "-qm", "fixture")
    return CommittedCatalog.from_git(tmp_path, ReadPolicy(allowed_prefixes=("src",)), RetrievalBudget())


@pytest.fixture
def scope():
    return InvestigationScope(task_id="T1", objective="Investigate orders",
                              criteria=(Criterion(criterion_id="C01", question="API contract", required_paths=(API,)),
                                        Criterion(criterion_id="C02", question="Mock behavior", required_paths=(MOCK,))))


class Backend:
    """Isolate session/runtime from search ranking. Paths still use real Git blobs."""
    def __init__(self, catalog, *, budget=None, hits=None):
        self.catalog = catalog
        self.budget = budget or RetrievalBudget()
        self.hits = hits
        self.semantic = None
        self.diagnostics = []

    def reset_runtime_caches(self):
        self.diagnostics.clear()

    async def find(self, gap):
        if self.hits is not None:
            return self.hits
        return [Candidate(path=p, line=1, end_line=len(self.catalog.files[p].splitlines()), channel="path")
                for p in authorized_path_matches(gap.query, self.catalog.files)]


class FakeLLM:
    def __init__(self, replies=()):
        self.replies = iter(replies)
        self.requests = []

    async def chat(self, request):
        self.requests.append(request.model_copy(deep=True))
        return next(self.replies)


def call(id, name, **args):
    return ToolCall(id=id, name=name, arguments=args)


def reply(*calls, content="", tokens=20, **kwargs):
    return ChatResponse(content=content, tool_calls=list(calls), total_tokens=tokens, **kwargs)


def lookup(id="lookup", **kwargs):
    return call(id, "retrieve_evidence", criterion_id="C02", query="src/mocks/service",
                strategy="path", **kwargs)


def report(mock_id="E001", unresolved=False):
    mock = {"criterion_id":"C02", "status":"resolved", "claim":"returns empty list",
            "citations":[{"evidence_id":mock_id,"path":MOCK,"line":2,"quote":"return []"}]}
    if unresolved:
        mock = {"criterion_id":"C02","status":"unresolved","claim":"not read","citations":[],
                "gap":{"criterion_id":"C02","question":"Mock behavior","reason":"missing source","query":"mockListOrders","strategy":"symbol"}}
    return json.dumps({"task_id":"T1", "summary":"investigation", "answers":[
        {"criterion_id":"C01","status":"resolved","claim":"calls mock","citations":[
            {"evidence_id":"S001","path":API,"line":2,"quote":"listOrders = () => mockListOrders()"}]}, mock]})


def run(catalog, scope, replies, *, config=None, **kwargs):
    llm = FakeLLM(replies)
    result = asyncio.run(AgenticRetrievalController(llm=llm, runtime_config=config).run(
        backend=Backend(catalog), scope=scope, initial_paths=(API,), **kwargs))
    return result, llm


@pytest.mark.parametrize("query,expected", [("src/mocks/service", [MOCK]), ("src/mocks/service.ts mockListOrders", [MOCK]),
    ("../src/mocks/service", []), ("/src/mocks/service", []), ("C:/src/mocks/service", []),
    ("src\\mocks\\service", []), ("src/.env.local", [])])
def test_authorized_path_normalization(catalog, query, expected):
    assert authorized_path_matches(query, catalog.files) == expected


def test_path_ambiguity_and_index():
    paths = {"src/a.ts", "src/a.js", "src/a/index.ts"}
    assert authorized_path_matches("src/a", paths) == sorted(paths)
    assert authorized_path_matches("src/a.ts", paths) == ["src/a.ts"]


def test_identifiers_and_minimal_request():
    assert identifiers_in("调查 Order interface status type Order") == ["Order"]
    assert RetrievalRequest(criterion_id="C02", query="mockListOrders").question is None
    with pytest.raises(ValueError):
        RetrievalRequest(criterion_id="C02", query="./service", strategy="dependency")


def test_reuse_seed_without_source_retransmission(catalog, scope):
    session = RetrievalSession(Backend(catalog), scope, (API,), default_runtime_config())
    before = session.used_chars
    data = session.read(ReadWindowArgs(path=API))
    assert data["read_status"] == "already_covered" and data["existing_evidence_ids"] == ["S001"]
    assert "lines" not in data and session.used_chars == before and not session.delivered


def test_partial_reuse_only_missing_lines(catalog, scope):
    session = RetrievalSession(Backend(catalog), scope, (API,), default_runtime_config(), seed_ranges={API: [(1, 1)]})
    data = session.read(ReadWindowArgs(path=API, start_line=1, end_line=2))
    assert data["start_line"] == data["end_line"] == 2
    assert data["existing_evidence_ids"] == ["S001"]
    assert data["registered_evidence_id"] == "E001"
    assert session.read(ReadWindowArgs(path=API))["existing_evidence_ids"] == ["S001", "E001"]


def test_combined_read_validated_and_independent_finalization(catalog, scope):
    config = default_runtime_config().model_copy(update={"max_steps":2})
    result, llm = run(catalog, scope, [reply(lookup(read_if_unique=True)), reply(content=report())], config=config)
    assert result.runtime.status == "completed", result.runtime.model_dump()
    assert result.runtime.tool_calls == 1
    assert result.runtime.observations[0].evidence_id == "E001"
    assert result.report is not None
    assert set(result.evidence_paths) == {API, MOCK}
    assert llm.requests[-1].tools is None
    assert len(llm.requests[-1].input_items) == 2
    assert {w["evidence_id"] for w in json.loads(llm.requests[-1].input_items[1]["content"])["seed_evidence"]} == {"S001", "E001"}
    assert all(item.get("type") != "function_call_output" for item in llm.requests[-1].input_items)


def test_combined_read_ambiguous_not_auto_read(catalog, scope):
    backend = Backend(catalog, budget=RetrievalBudget(max_candidates_per_call=1), hits=[
        Candidate(path=MOCK, line=1, end_line=3, channel="path"),
        Candidate(path=API, line=1, end_line=2, channel="path")])
    session = RetrievalSession(backend, scope, (), default_runtime_config())
    result = asyncio.run(session.retrieve(EvidenceGap(criterion_id="C02", question="mock", reason="missing", query="service"), read_if_unique=True))
    assert len(result["candidates"]) == 1 and "evidence" not in result and not session.delivered


def test_combined_evidence_forged_source_rejected(catalog, scope):
    session = RetrievalSession(Backend(catalog), scope, (API,), default_runtime_config())
    data = asyncio.run(session.retrieve(EvidenceGap(criterion_id="C02", question="mock", reason="missing", query=MOCK, strategy="path"), read_if_unique=True))
    data["evidence"]["lines"][1] = "  return ['fake']"
    data["evidence"]["numbered_lines"][1]["content"] = "  return ['fake']"
    obs = Observation(call_id="c", tool_name="retrieve_evidence", success=True, data=data, evidence_id="E001")
    with pytest.raises(EvidenceError):
        EvidenceLedger.from_observations(catalog, session.seeds, [obs])


def test_missing_required_path_zero_model_calls(catalog, scope):
    restricted = CommittedCatalog(catalog.revision, {API: catalog.files[API]})
    result, llm = run(restricted, scope, [])
    assert result.runtime.reason == "required_evidence_unavailable"
    assert not llm.requests and result.runtime.llm_attempts == 0 and result.runtime.reported_tokens == 0
    assert result.report is None


def test_disable_preflight_does_not_disable_gate(catalog, scope):
    restricted = CommittedCatalog(catalog.revision, {API: catalog.files[API]})
    result, llm = run(restricted, scope, [reply(content=report(unresolved=True))], scope_preflight=False)
    assert result.runtime.status == "blocked" and result.validation_code == "unresolved_criteria"
    assert result.runtime.candidate_output and result.runtime.final_output is None


def test_token_budget_blocks_before_unaffordable_request(catalog, scope):
    result, llm = run(catalog, scope, [], config=default_runtime_config().model_copy(update={"max_total_tokens":100}))
    assert result.runtime.reason == "token_budget_exhausted" and not llm.requests


def test_response_usage_still_authoritative(catalog, scope):
    result, llm = run(catalog, scope, [reply(content=report(), tokens=40001)])
    assert result.runtime.reason == "token_budget_exhausted" and result.runtime.reported_tokens == 40001
    assert result.runtime.final_output is None and result.runtime.candidate_output is not None


def test_incomplete_response_keeps_usage_diagnostics(catalog, scope):
    result, _ = run(catalog, scope, [reply(content="partial", tokens=2000, prompt_tokens=1500,
                                         completion_tokens=500, response_status="incomplete")], mode="fixed_context")
    event = next(e for e in result.runtime.trace if e.event == "llm_completed")
    assert event.data["prompt_tokens"] == 1500 and event.data["completion_tokens"] == 500
    assert result.runtime.reason == "incomplete_model_response" and result.runtime.candidate_output == "partial"


def test_budget_estimation_calibrates_upward():
    assert estimated_input_tokens(20000, .5) > estimated_input_tokens(20000, .2)


def test_finalization_rejects_provider_tool_calls(catalog, scope):
    config = default_runtime_config().model_copy(update={"max_steps":1})
    result, _ = run(catalog, scope, [reply(lookup(read_if_unique=True))], config=config)
    assert result.runtime.reason == "tool_calls_during_finalization" and not result.runtime.observations


def test_seed_only_success_unchanged(catalog, scope):
    llm = FakeLLM([reply(content=report(mock_id="S002"))])
    result = asyncio.run(AgenticRetrievalController(llm=llm).run(backend=Backend(catalog), scope=scope, initial_paths=(API, MOCK)))
    assert result.runtime.status == "completed" and result.runtime.tool_calls == 0


def test_budget_reserve_forces_report_before_last_step(catalog, scope):
    result, llm = run(catalog, scope, [reply(lookup(read_if_unique=True), tokens=33000),
                                      reply(content=report(), tokens=2000)])
    assert result.runtime.status == "completed", result.runtime.model_dump()
    assert len(llm.requests) == 2 and llm.requests[-1].tools is None
    assert any(e.event == "finalization_started" and e.data["reason"] == "finalization_token_reserve"
               for e in result.runtime.trace)
    assert result.runtime.reported_tokens == 35000


def test_symbol_query_normalization_with_indexed_identifier(catalog):
    backend = RetrievalBackend.__new__(RetrievalBackend)
    backend.catalog = catalog
    backend.budget = RetrievalBudget()
    backend.diagnostics = []
    backend._symbols = lambda gap: [Candidate(path=MOCK, line=1, end_line=3, channel="symbol")] if gap.query == "mockListOrders" else []
    gap = EvidenceGap(criterion_id="C02", question="mock", reason="missing", strategy="symbol", query="mockListOrders function definition")
    found = asyncio.run(backend.find(gap))
    assert [item.path for item in found] == [MOCK]
    assert "query_normalization_used" in backend.diagnostics


def test_initial_retrieval_fills_empty_baseline_with_task_symbols(catalog, monkeypatch):
    from agent_core.agentic_retrieval import retrieval
    backend = RetrievalBackend.__new__(RetrievalBackend)
    backend.catalog = catalog
    backend.budget = RetrievalBudget()
    backend.index = object()
    backend.exact = object()
    backend.hybrid = object()
    backend.semantic = None
    backend._symbols = lambda gap: [Candidate(path=MOCK, channel="symbol")] if gap.query == "mockListOrders" else []

    class FakeService:
        def __init__(self, **kwargs):
            pass
        def build_query_specs(self, **kwargs):
            return [SimpleNamespace(text="中文调查")]
        async def retrieve(self, **kwargs):
            assert kwargs["semantic_enabled"] is False
            return SimpleNamespace(selected_files=[])

    monkeypatch.setattr(retrieval, "TaskRetrievalService", FakeService)
    task = SimpleNamespace(objective="调查 mockListOrders", investigation_questions=["确认 mockListOrders"])
    paths = asyncio.run(backend.initial_candidates(task=task))
    assert paths == (MOCK,)
    assert backend.initial_diagnostics[0]["baseline_paths"] == []
    assert backend.initial_diagnostics[0]["selected_paths"] == [MOCK]


def test_provider_incomplete_reason_and_reasoning_tokens_mapped():
    # The SDK client is replaced by a scripted response, no network or key.
    from agent_core.llm.openai import OpenAILLM
    from agent_core.llm.model import ChatRequest, Message

    class Responses:
        async def create(self, **kwargs):
            return SimpleNamespace(output=[], output_text="partial", status="incomplete", model="fake",
                                   incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                                   usage=SimpleNamespace(input_tokens=100, output_tokens=200,
                                                         total_tokens=300,
                                                         output_tokens_details=SimpleNamespace(reasoning_tokens=150)))
    adapter = object.__new__(OpenAILLM)
    adapter.model = "fake"
    adapter.client = SimpleNamespace(responses=Responses())
    result = asyncio.run(adapter.chat(ChatRequest(messages=[Message(role="user", content="test")])))
    assert result.incomplete_reason == "max_output_tokens"
    assert result.reasoning_tokens == 150
    assert result.prompt_tokens == 100 and result.completion_tokens == 200 and result.total_tokens == 300
