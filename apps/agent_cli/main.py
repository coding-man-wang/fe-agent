import asyncio
import os

from dotenv import load_dotenv

from agent_core.llm.openai import OpenAILLM
from agent_core.sdd.requirement import RequirementAnalyzer
from agent_core.sdd.validation import RequireValidator

load_dotenv()


async def main():
    llm = OpenAILLM(
        apikey=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL_ID"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    # list_file_tool = ListFileTool()
    # tool_registry = ToolRegistry()
    # tool_registry.register(list_file_tool)
    # config = AgentConfig()
    # agent = Agent(
    #     llm=llm,
    #     tool_registry=tool_registry,
    #     config=config,
    #     retry_policy=None
    # )
    #
    # await agent.run("请查看当前项目有哪些文件")

    analyzer = RequirementAnalyzer(
        llm=llm
    )

    spec = await analyzer.analyze(
        """
        给现有 Vue3 管理后台增加用户列表页面。

        要求：
        - 支持用户名搜索
        - 支持分页
        - 必须使用公司的 Table 组件
        """
    )

    validator = RequireValidator(
        llm=llm
    )

    validation = await validator.validate(
        spec
    )
    print(spec)
    print(
        spec.model_dump_json(
            indent=2
        )
    )

    print(
        validation.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
