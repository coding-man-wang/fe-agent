import json
from typing import Any

from agent_core.llm.model import ChatResponse


def replay_items(response: ChatResponse) -> list[dict[str, Any]]:
    """Keep provider output (including reasoning items) for stateless replay."""
    if response.output_items is not None:
        items = response.output_items
        actual = [item for item in items if item.get('type') == 'function_call']
        if len(actual) != len(response.tool_calls):
            raise ValueError('provider tool call replay mismatch')
        for item, call in zip(actual, response.tool_calls):
            if (item.get('call_id') != call.id or item.get('name') != call.name
                    or json.loads(item['arguments']) != call.arguments):
                raise ValueError('provider tool call replay mismatch')
        return items
    # Fake or non-reasoning adapters may omit raw items.
    items = []
    if response.content:
        items.append({'role': 'assistant', 'content': response.content})
    items.extend({'type': 'function_call', 'call_id': c.id, 'name': c.name,
                  'arguments': json.dumps(c.arguments, ensure_ascii=False)}
                 for c in response.tool_calls)
    return items


def function_output(call_id: str, content: str) -> dict[str, str]:
    return {'type': 'function_call_output', 'call_id': call_id, 'output': content}
