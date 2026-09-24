import asyncio
import os
from dotenv import load_dotenv
from agent_core.llm.model import ChatRequest, Message
from agent_core.tools.builtin.list_files import ListFileTool
from agent_core.llm.openai import OpenAILLM

load_dotenv()


async def main():
    llm = OpenAILLM(
        apikey=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL_ID"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    tool = ListFileTool()
    print(tool.to_schema())
    request = ChatRequest(
        messages=[Message(role="user", content="帮我查看当前目录有哪些文件")],
        tools=[tool.to_schema()],
    )

    response = await llm.chat(request)

    print(response)


if __name__ == "__main__":
    asyncio.run(main())
