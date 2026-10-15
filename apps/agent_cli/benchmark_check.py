import asyncio

from agent_core.benchmark.case_loader import (
    BenchmarkCaseLoader,
)
from agent_core.benchmark.repository_guard import (
    BenchmarkRepositoryGuard,
)


async def main():

    loader = BenchmarkCaseLoader("benchmark/cases")

    case = loader.load("CASE-001")

    guard = BenchmarkRepositoryGuard(
        repository_root=(r"D:\project\fe-agent-benchmark-web")
    )

    state = await guard.require_ready(case=case)

    print(state.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
