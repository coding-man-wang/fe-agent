import asyncio
import os

from dotenv import load_dotenv

from agent_core.agent.agent import Agent
from agent_core.agent.config import AgentConfig
from agent_core.llm.openai import OpenAILLM
from agent_core.tools.builtin.list_files import ListFileTool
from agent_core.tools.registry import ToolRegistry

load_dotenv()


async def main():
    llm = OpenAILLM(
        apikey=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL_ID"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    list_file_tool = ListFileTool()
    tool_registry = ToolRegistry()
    tool_registry.register(list_file_tool)
    config = AgentConfig()
    agent = Agent(
        llm=llm,
        tool_registry=tool_registry,
        config=config,
        retry_policy=None
    )

    await agent.run("请查看当前项目有哪些文件")


if __name__ == "__main__":
    asyncio.run(main())
