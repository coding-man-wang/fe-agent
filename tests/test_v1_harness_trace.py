from agent_core.harness.trace import HarnessTraceEvent, JsonlHarnessTraceStore


def test_trace_round_trip(tmp_path):
    store=JsonlHarnessTraceStore(path=tmp_path/"trace.jsonl")
    store.append(HarnessTraceEvent(task_id="TASK-001",event_type="investigation_recorded",data={"n":2}))
    events=store.read_all()
    assert len(events)==1
    assert events[0].data["n"]==2
