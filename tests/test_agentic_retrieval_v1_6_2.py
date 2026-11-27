"""v1.6.2 report-boundary regressions. No provider network required."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from test_agentic_retrieval_v1_6_1 import (catalog, scope, run, reply, lookup, report,
    Backend, FakeLLM, API, MOCK, call)
from agent_core.agentic_retrieval.contracts import CitationReference, InvestigationDraft
from agent_core.agentic_retrieval.controller import AgenticRetrievalController, default_runtime_config
from agent_core.agentic_retrieval.evidence import CompletionGate, EvidenceLedger, EvidenceWindow, EvidenceError
from agent_core.llm.model import ChatRequest, Message


def draft(mock_id="E001"):
    data = json.loads(report(mock_id=mock_id))
    for answer in data["answers"]:
        for citation in answer["citations"]:
            citation.pop("path")
            citation.pop("quote")
    return data


def test_coordinate_only_report_is_canonical(catalog, scope):
    result, llm = run(catalog, scope, [reply(lookup(read_if_unique=True)), reply(content=json.dumps(draft()))],
                      config=default_runtime_config().model_copy(update={"max_steps":2}))
    assert result.runtime.status == "completed", result.runtime.model_dump()
    assert result.report.answers[1].citations[0].path == MOCK
    assert result.report.answers[1].citations[0].quote == "  return []"
    assert json.loads(result.runtime.final_output) == result.report.model_dump(mode="json")
    assert result.runtime.candidate_output is None
    assert llm.requests[0].max_output_tokens == 2048
    assert llm.requests[1].max_output_tokens == 8192
    assert llm.requests[1].tools is None
    assert len(llm.requests[1].input_items) == 2


@pytest.mark.parametrize("changes", [{"evidence_id":"E999"}, {"line":999}, {"end_line":999}, {"line":0}, {"line":True}, {"path":MOCK}, {"quote":"fake"}])
def test_invalid_draft_citation_rejected(catalog, scope, changes):
    data = draft("S002")
    data["answers"][1]["citations"][0].update(changes)
    llm = FakeLLM([reply(content=json.dumps(data))])
    result = asyncio.run(AgenticRetrievalController(llm=llm).run(backend=Backend(catalog), scope=scope, initial_paths=(API, MOCK), mode="fixed_context"))
    assert result.runtime.status == "blocked"
    assert result.report is None and result.runtime.final_output is None
    assert result.runtime.candidate_output


def test_legacy_forged_quote_not_materialized(catalog, scope):
    data = json.loads(report(mock_id="S002"))
    data["answers"][1]["citations"][0]["quote"] = "fake code"
    seeds = (EvidenceWindow.from_catalog(catalog, "S001", API, 1, 2), EvidenceWindow.from_catalog(catalog, "S002", MOCK, 1, 3))
    with pytest.raises(EvidenceError, match="quote"):
        CompletionGate(scope, catalog, seeds).validate(json.dumps(data), [])


def test_multiline_exact_materialization(catalog):
    ledger = EvidenceLedger(catalog, [EvidenceWindow.from_catalog(catalog,"S001",MOCK,1,3)])
    citation = ledger.materialize(CitationReference(evidence_id="S001",line=1,end_line=3))
    assert citation.quote == catalog.files[MOCK].rstrip("\n")
    assert ledger.validate(citation).path == MOCK


def test_reference_cannot_cross_window_boundary(catalog):
    ledger = EvidenceLedger(catalog, [EvidenceWindow.from_catalog(catalog,"S001",MOCK,1,1), EvidenceWindow.from_catalog(catalog,"S002",MOCK,2,3)])
    with pytest.raises(EvidenceError):
        ledger.materialize(CitationReference(evidence_id="S001",line=1,end_line=2))


def test_required_path_gate_stays_strict(catalog, scope):
    data = draft("S001")
    data["answers"][1]["citations"][0]["line"] = 2
    seeds = (EvidenceWindow.from_catalog(catalog,"S001",API,1,2),)
    with pytest.raises(EvidenceError) as err:
        CompletionGate(scope,catalog,seeds).validate(json.dumps(data),[])
    assert err.value.code == "required_evidence_missing"


def test_roles_cannot_acquire_new_write_permission(catalog, scope):
    data = draft("S002")
    data["observed_files"] = [{"relative_path":MOCK,"observed_role":"candidate_target","rationale":"mock behavior", "confidence":0.9,"criterion_ids":["C02"],"citations":[{"evidence_id":"S002","line":2}]}]
    seeds = (EvidenceWindow.from_catalog(catalog,"S001",API,1,2), EvidenceWindow.from_catalog(catalog,"S002",MOCK,1,3))
    with pytest.raises(EvidenceError) as err:
        CompletionGate(scope,catalog,seeds,candidate_target_ceiling=frozenset({API})).validate(json.dumps(data),[])
    assert err.value.code == "new_target_requires_approval"


def test_reserve_accounts_for_full_final_cap(catalog, scope):
    result,llm = run(catalog,scope,[reply(lookup(read_if_unique=True),tokens=30000), reply(content=json.dumps(draft()),tokens=2000)])
    assert result.runtime.status == "completed"
    assert llm.requests[1].tools is None
    event = next(e for e in result.runtime.trace if e.event=="llm_started")
    assert event.data["finalization_reserve_tokens"] > 8192
    assert result.runtime.reported_tokens == 32000
    assert any(e.event=="finalization_started" and e.data["reason"]=="finalization_token_reserve" for e in result.runtime.trace)


def test_final_truncation_not_accepted(catalog, scope):
    result,llm = run(catalog,scope,[reply(content="partial",response_status="incomplete",incomplete_reason="max_output_tokens")],mode="fixed_context")
    assert result.runtime.reason=="incomplete_model_response"
    assert result.runtime.candidate_output=="partial" and result.runtime.final_output is None
    assert len(llm.requests)==1


def test_truncated_exploration_text_gets_one_final_request(catalog, scope):
    llm=FakeLLM([reply(content="partial",response_status="incomplete",incomplete_reason="max_output_tokens"),reply(content=json.dumps(draft("S002")))])
    result=asyncio.run(AgenticRetrievalController(llm=llm).run(backend=Backend(catalog),scope=scope,initial_paths=(API,MOCK)))
    assert result.runtime.status=="completed"
    assert result.runtime.candidate_output is None
    assert len(llm.requests)==2 and llm.requests[1].tools is None
    assert any(e.event=="finalization_started" and e.data["reason"]=="exploration_output_exhausted" for e in result.runtime.trace)


def test_incomplete_tool_batch_never_dispatched(catalog, scope):
    result,llm=run(catalog,scope,[reply(lookup(read_if_unique=True),response_status="incomplete",incomplete_reason="max_output_tokens")])
    assert result.runtime.status=="blocked" and not result.runtime.observations
    assert result.runtime.tool_calls==0 and len(llm.requests)==1


def test_unknown_tool_fields_logged_without_values(catalog, scope):
    result,_=run(catalog,scope,[reply(lookup(extra_field="secret-value",**{"bad field":"secret-value"})),reply(content=json.dumps(draft()))],config=default_runtime_config().model_copy(update={"max_steps":2}))
    event=next(e for e in result.runtime.trace if e.event=="unknown_tool_fields")
    assert event.data["unknown_fields"]==["extra_field"] and event.data["unknown_field_count"]==2
    assert "secret-value" not in json.dumps(event.data)
    assert result.runtime.status=="blocked"


@pytest.mark.parametrize("effort", [None,"low"])
def test_reasoning_effort_forwarded_only_opt_in(effort):
    from agent_core.llm.openai import OpenAILLM
    captured=[]
    class Responses:
        async def create(self,**kwargs):
            captured.append(kwargs)
            return SimpleNamespace(output=[],output_text="ok",status="completed",model="fake",usage=None)
    adapter=object.__new__(OpenAILLM)
    adapter.model="fake"
    adapter.client=SimpleNamespace(responses=Responses())
    asyncio.run(adapter.chat(ChatRequest(messages=[Message(role="user",content="test")],reasoning_effort=effort)))
    assert captured[0].get("reasoning") == ({"effort":"low"} if effort else None)


def test_draft_schema_uses_coordinates_only():
    schema=InvestigationDraft.model_json_schema()["$defs"]["CitationReference"]
    assert set(schema["properties"])=={"evidence_id","line","end_line"}
    assert schema["additionalProperties"] is False
