class AgentError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        self.retryable = retryable
        super().__init__(message)


class MaxStepsError(AgentError):
    def __init__(self, max_steps: int):
        self.max_steps = max_steps

        super().__init__(f"Agent exceeded max steps: {max_steps}")


class ToolNotFoundError(AgentError):
    def __init__(self, tool_name: str):
        self.tool_name = tool_name

        super().__init__(f"Tool not found: {tool_name}")


class ToolValidationError(AgentError):
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name

        super().__init__(f"Invalid arguments for tool '{tool_name}': {message}")


class ToolExecutionError(AgentError):
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name

        super().__init__(f"Tool '{tool_name}' execution failed: {message}")


class LLMError(AgentError):
    def __init__(
        self, message: str, retryable: bool = False, status_code: int | None = None
    ):
        self.status_code = status_code
        super().__init__(f"LLM request failed: {message}", retryable=retryable)
