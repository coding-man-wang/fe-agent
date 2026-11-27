import asyncio
import json
from time import perf_counter
from typing import Any

from pydantic import ValidationError
from agent_core.agent.model import ToolCall
from agent_core.tools.registry import ToolRegistry
from .model import Observation, RuntimeConfig


class ToolDispatcher:
    """Only explicitly trusted read-only tools may enter this runtime.

    The allowlist is application-owned, not supplied by the model. Tool code is
    trusted Python; this is not a sandbox for malicious third-party plugins.
    """
    def __init__(self, registry: ToolRegistry, *, allowed_names: set[str],
                 config: RuntimeConfig):
        tools = {t.name: t for t in registry.list_tools()}
        if not allowed_names <= tools.keys():
            raise ValueError('allowlist contains unregistered tools')
        self._tools = {name: tools[name] for name in sorted(allowed_names)}
        self.config = config

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute_batch(self, calls: list[ToolCall]) -> list[Observation]:
        semaphore = asyncio.Semaphore(self.config.max_parallel_tools)

        async def execute(call: ToolCall) -> Observation:
            async with semaphore:
                started = perf_counter()

                def result(success: bool, data=None, error=None):
                    return Observation(call_id=call.id, tool_name=call.name,
                                       success=success, data=data, error_code=error,
                                       duration_ms=round((perf_counter()-started)*1000, 2))
                tool = self._tools.get(call.name)
                if tool is None:
                    return result(False, error='tool_not_allowed')
                try:
                    if len(json.dumps(call.arguments, ensure_ascii=False).encode()) > self.config.max_argument_bytes:
                        return result(False, error='arguments_too_large')
                    # Pydantic models can default to ignoring extra fields; reject
                    # them here even for an older tool schema.
                    if set(call.arguments) - set(tool.args_schema.model_fields):
                        return result(False, error='invalid_arguments')
                    args = tool.args_schema.model_validate(call.arguments, strict=True)
                except (ValidationError, ValueError, TypeError):
                    return result(False, error='invalid_arguments')
                try:
                    async with asyncio.timeout(self.config.tool_timeout_seconds):
                        data = await tool.execute(args.model_dump(mode='json'))
                    if isinstance(data, dict) and data.get('success') is False:
                        return result(False, error='tool_reported_failure')
                    # Serialize now: invalid or huge output must not enter context.
                    # Normalize Pydantic or ordinary JSON-compatible tool results.
                    if hasattr(data, 'model_dump'):
                        data = data.model_dump(mode='json')
                    data = json.loads(json.dumps(data, ensure_ascii=False, allow_nan=False))
                    observation = result(True, data=data)
                    if len(observation.output().encode()) > self.config.max_tool_output_bytes:
                        return result(False, error='tool_output_too_large')
                    return observation
                except TimeoutError:
                    return result(False, error='tool_timeout')
                except Exception:
                    # Do not disclose host paths, secrets, or exception internals.
                    return result(False, error='tool_execution_failed')

        # gather preserves request order, not completion order.
        return list(await asyncio.gather(*(execute(call) for call in calls)))
