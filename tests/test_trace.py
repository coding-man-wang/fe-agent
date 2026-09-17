import pytest

from agent_core.agent.state import AgentState


def test_add_trace():
    state = AgentState()

    state.add_trace(
        "run_started"
    )

    assert len(state.trace) == 1

    event = state.trace[0]

    assert (
            event.run_id
            == state.run_id
    )

    assert (
            event.event_type
            == "run_started"
    )

    print(event.event_type)
    assert event.step == 0


def test_trace_data():
    state = AgentState()

    state.step = 2

    state.add_trace(
        "tool_started",
        data={
            "tool_name": "list_file"
        }
    )

    event = state.trace[0]

    assert event.step == 2

    assert (
            event.data["tool_name"]
            == "list_file"
    )


def test_trace_event_has_unique_id():
    state = AgentState()

    state.add_trace(
        "run_started"
    )

    state.add_trace(
        "run_completed"
    )

    assert (
            state.trace[0].event_id
            !=
            state.trace[1].event_id
    )


if __name__ == "__main__":
    pytest.main([__file__])
