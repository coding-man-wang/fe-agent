import asyncio

from agent_core.tools.builtin.list_files import ListFileTool
from agent_core.tools.registry import ToolRegistry

async def main():
    registry = ToolRegistry()

    registry.register(ListFileTool())

    list_file_tool = registry.get("list_file")

    print(list_file_tool.to_schema())

    result = await list_file_tool.execute({
        "xxx": "."
    })

    print(result)

if __name__ == "__main__":

    asyncio.run(main())
