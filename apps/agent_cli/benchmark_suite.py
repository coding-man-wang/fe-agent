import asyncio
from pathlib import Path

from dotenv import load_dotenv

from agent_core.benchmark.case_loader import (
    BenchmarkCaseLoader,
)
from agent_core.benchmark.result_builder import (
    BenchmarkResultBuilder,
)
from agent_core.benchmark.result_store import (
    BenchmarkResultStore,
)
from agent_core.benchmark.suite_report import (
    BenchmarkSuiteTextReportRenderer,
)
from agent_core.benchmark.suite_runner import (
    BenchmarkSuiteRunner,
)
from agent_core.benchmark.suite_store import (
    BenchmarkSuiteStore,
)
from apps.agent_cli.benchmark_factory import (
    create_benchmark_runner,
)

load_dotenv()


async def main():

    fe_agent_root = Path(__file__).resolve().parents[2]

    benchmark_repository = Path(r"D:\project\fe-agent-benchmark-web")

    case_loader = BenchmarkCaseLoader(fe_agent_root / "benchmark" / "cases")

    benchmark_runner = create_benchmark_runner(
        fe_agent_root=(fe_agent_root),
        benchmark_repository=(benchmark_repository),
    )

    suite_runner = BenchmarkSuiteRunner(
        case_loader=(case_loader),
        benchmark_runner=(benchmark_runner),
        result_builder=(BenchmarkResultBuilder()),
        result_store=(BenchmarkResultStore(fe_agent_root)),
    )

    suite = await suite_runner.run_all()

    suite_store = BenchmarkSuiteStore(fe_agent_root)

    suite_store.save(suite)

    report = BenchmarkSuiteTextReportRenderer().render(suite)

    print(report)


if __name__ == "__main__":

    asyncio.run(main())
