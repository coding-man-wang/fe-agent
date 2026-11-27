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

    def __init__(
        self, registry: ToolRegistry, *, allowed_names: set[str], config: RuntimeConfig
    ):
        tools = {t.name: t for t in registry.list_tools()}
        if not allowed_names <= tools.keys():
            raise ValueError("allowlist contains unregistered tools")
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
                    return Observation(
                        call_id=call.id,
                        tool_name=call.name,
                        success=success,
                        data=data,
                        error_code=error,
                        duration_ms=round((perf_counter() - started) * 1000, 2),
                    )

                tool = self._tools.get(call.name)
                if tool is None:
                    return result(False, error="tool_not_allowed")
                try:

                    argument_bytes = len(
                        json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                        ).encode()
                    )

                    if argument_bytes > self.config.max_argument_bytes:
                        return result(
                            False,
                            data={
                                "message": "Tool arguments exceed size limit.",
                                "max_argument_bytes": (self.config.max_argument_bytes),
                            },
                            error="arguments_too_large",
                        )

                    # ==================================
                    # Reject Unknown Fields
                    # ==================================

                    unknown_fields = set(call.arguments) - set(
                        tool.args_schema.model_fields
                    )

                    if unknown_fields:
                        return result(
                            False,
                            data={
                                "message": ("Unexpected argument fields."),
                                "allowed_fields": sorted(tool.args_schema.model_fields),
                            },
                            error="invalid_arguments",
                        )

                    # ==================================
                    # Schema Validation
                    # ==================================

                    args = tool.args_schema.model_validate(
                        call.arguments,
                        strict=True,
                    )

                except ValidationError as exc:

                    errors = []

                    allowed_fields = set(tool.args_schema.model_fields)

                    for item in exc.errors(
                        include_input=False,
                        include_context=False,
                        include_url=False,
                    )[:5]:
                        location = item.get("loc", ())

                        field = (
                            location[0]
                            if location and location[0] in allowed_fields
                            else "__root__"
                        )

                        errors.append(
                            {
                                "field": field,
                                "type": item["type"],
                            }
                        )

                    data = {
                        "message": (
                            "Arguments do not satisfy tool schema. "
                            "Correct the reported fields before retrying."
                        ),
                        "errors": errors,
                    }

                    # read_file 存在跨字段约束，
                    # 需要将该约束明确反馈给模型。
                    if call.name == "read_file":
                        data["range_rule"] = (
                            "start_line >= 1; "
                            "end_line >= start_line; "
                            "end_line - start_line < 100. "
                            "For small files, omit both line arguments."
                        )

                    return result(
                        False,
                        data=data,
                        error="invalid_arguments",
                    )

                except (ValueError, TypeError):

                    return result(
                        False,
                        data={
                            "message": "Invalid tool arguments.",
                        },
                        error="invalid_arguments",
                    )
                try:
                    async with asyncio.timeout(self.config.tool_timeout_seconds):
                        data = await tool.execute(args.model_dump(mode="json"))
                    if isinstance(data, dict) and data.get("success") is False:
                        return result(False, error="tool_reported_failure")
                    # Serialize now: invalid or huge output must not enter context.
                    # Normalize Pydantic or ordinary JSON-compatible tool results.
                    if hasattr(data, "model_dump"):
                        data = data.model_dump(mode="json")
                    data = json.loads(
                        json.dumps(data, ensure_ascii=False, allow_nan=False)
                    )
                    observation = result(True, data=data)
                    if (
                        len(observation.output().encode())
                        > self.config.max_tool_output_bytes
                    ):
                        return result(False, error="tool_output_too_large")
                    return observation
                except TimeoutError:
                    return result(False, error="tool_timeout")
                except Exception:
                    # Do not disclose host paths, secrets, or exception internals.
                    return result(False, error="tool_execution_failed")

        # gather preserves request order, not completion order.
        return list(await asyncio.gather(*(execute(call) for call in calls)))
