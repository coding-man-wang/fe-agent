import asyncio
from pathlib import Path

from dotenv import load_dotenv

from apps.agent_cli.benchmark_factory import (
    create_benchmark_runner,
)

load_dotenv()


async def main():

    fe_agent_root = Path(__file__).resolve().parents[2]

    benchmark_repository = Path(r"D:\project\fe-agent-benchmark-web")

    runner = create_benchmark_runner(
        fe_agent_root=(fe_agent_root),
        benchmark_repository=(benchmark_repository),
    )

    result = await runner.run(case_id="CASE-004")

    print(
        result.model_dump_json(
            indent=2,
        )
    )


if __name__ == "__main__":

    asyncio.run(main())
