import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agent_core.coding.agent import (
    ReadOnlyCodingAgent,
)
from agent_core.coding.llm_adapter import (
    OpenAITextModel,
)
from tests.test_context_renderer import (
    create_test_bundle,
)

load_dotenv()

@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("LLM_API_KEY"),
    reason="未配置真实模型 API Key",
)
async def test_live_readonly_analysis(
    tmp_path: Path,
):
    client = AsyncOpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.getenv("LLM_BASE_URL"),
    )

    model = OpenAITextModel(
        client=client,
        model=os.getenv(
            "LLM_MODEL_ID",
            "gpt-4.1-mini",
        ),
    )

    agent = ReadOnlyCodingAgent(model=model)

    bundle = create_test_bundle(tmp_path)

    result = await agent.analyze(bundle)

    assert result.task_id == "TASK-003"

    assert result.execution_status == "not_executed"

    print(result.model_dump_json(indent=2))
