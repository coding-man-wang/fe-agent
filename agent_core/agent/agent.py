from time import perf_counter

from agent_core.agent.config import AgentConfig
from agent_core.agent.errors import LLMError, MaxStepsError
from agent_core.agent.model import ToolResult, ToolCall
from agent_core.agent.retry import RetryPolicy, retry_async
from agent_core.agent.state import AgentState
from agent_core.llm.base import LLM
from agent_core.llm.model import ChatRequest, Message
from agent_core.tools.registry import ToolRegistry


class Agent:
    def __init__(self, llm: LLM, tool_registry: ToolRegistry,
                 config: AgentConfig, retry_policy: RetryPolicy):
        self.llm = llm
        self.tool_registry = tool_registry
        self.config = config or AgentConfig()
        self.retry_policy = retry_policy or RetryPolicy()

    async def _execute_tool(self, tool_call: ToolCall, state: AgentState):
        try:
            tool = self.tool_registry.get(tool_call.name)

            state.add_trace("tool_started", data={
                "tool_name": tool_call.name,
                "tool_call_id": tool_call.id,
            })

            try:
                tool_start_time = perf_counter()

                result = await tool.execute(tool_call.arguments)
            except Exception as e:
                state.add_trace("tool_failed", data={
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "error_type": type(e).__name__,
                    "message": str(e),
                })
                raise

            tool_duration_ms = round((perf_counter() - tool_start_time) * 1000, 2)

            state.add_trace("tool_completed", data={
                "tool_name": tool_call.name,
                "tool_call_id": tool_call.id,
                "duration_ms": tool_duration_ms,
            })

            print("Tool Result:", result, end="\n\n")

            tool_result = ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=str(result),
                success=True
            )

            return tool_result
        except Exception as e:
            tool_result = ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Tool execution failed: {
                str(e)}",
                success=False
            )
            return tool_result

    async def run(self, user_input: str):
        state = AgentState(messages=[
            Message(
                role="user",
                content=user_input,
            )
        ])

        state.status = "running"

        state.add_trace("run_started")

        try:
            step = 0

            while step < self.config.max_steps:

                state.step = step + 1

                state.add_trace(
                    "llm_started", data={
                        "model": getattr(
                            self.llm, "model", None)})

                print(f"\n===== Agent Step {step + 1} =====")
                step += 1

                try:
                    request = ChatRequest(
                        messages=state.messages,
                        tools=[
                            tool.to_schema()
                            for tool in self.tool_registry.list_tools()
                        ]
                    )

                    async def call_llm():
                        return await self.llm.chat(request)

                    llm_start_time = perf_counter()

                    def on_retry(retry_count, delay, e):
                        state.add_trace("retry_scheduled", data={
                            "retry_number": retry_count,
                            "delay": delay,
                            "error_type": type(e).__name__,
                        })

                    try:
                        response = await retry_async(call_llm, self.retry_policy, on_retry)
                    except Exception as e:
                        state.add_trace("llm_failed", data={
                            "error_type": type(e).__name__,
                            "message": str(e),
                        })
                        raise

                    llm_duration_ms = round(
                        (
                                perf_counter()
                                - llm_start_time
                        )
                        * 1000,
                        2,
                    )

                    state.add_trace(
                        "llm_completed",
                        data={
                            "model": response.model,
                            "duration_ms": llm_duration_ms,
                            "tool_call_count": len(
                                response.tool_calls)})
                except Exception as e:
                    raise LLMError(str(e)) from e

                print("LLM Response:", response)

                if not response.tool_calls:
                    state.status = "completed"

                    state.final_output = response.content

                    state.add_trace("run_completed")

                    for event in state.trace:
                        print(
                            event.model_dump()
                        )
                    return response

                state.messages.append(Message(
                    role="assistant",
                    content=response.content,
                ))

                for tool_call in response.tool_calls:
                    print("Tool Call:", tool_call)

                    tool_result = await self._execute_tool(tool_call, state)

                    state.messages.append(Message(
                        role="user",
                        content=str(tool_result.content),
                    ))
            raise MaxStepsError(self.config.max_steps)
        except Exception as e:
            state.status = "failed"

            state.error = str(e)

            state.add_trace("run_failed", data={
                "error_type": type(e).__name__,
                "message": str(e),
            })

            raise
