from agent_core.agent.state import AgentState


def test_agent_state_default():

    state = AgentState()

    assert state.run_id is not None

    assert state.status == "created"

    assert state.step == 0

    assert state.messages == []

def test_agent_state_isolated():

    state1 = AgentState()
    state2 = AgentState()

    assert state1.run_id != state2.run_id

    assert (
        state1.messages
        is not state2.messages
    )