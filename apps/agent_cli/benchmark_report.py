import sys
from pathlib import Path

from agent_core.benchmark.report import (
    BenchmarkTextReportRenderer,
)
from agent_core.benchmark.result_builder import (
    BenchmarkResultBuilder,
)
from agent_core.benchmark.result_store import (
    BenchmarkResultStore,
)
from agent_core.benchmark.run_store import (
    BenchmarkRunStore,
)


def main():

    if len(sys.argv) != 2:

        raise SystemExit(
            "Usage: "
            "uv run python "
            "apps/agent_cli/"
            "benchmark_report.py "
            "<run_id>"
        )

    run_id = sys.argv[1]

    fe_agent_root = Path(__file__).resolve().parents[2]

    run_store = BenchmarkRunStore(fe_agent_root)

    result_store = BenchmarkResultStore(fe_agent_root)

    run = run_store.load(run_id)

    result = BenchmarkResultBuilder().build(run=run)

    result_store.save(result)

    report = BenchmarkTextReportRenderer().render(result)

    print(report)


if __name__ == "__main__":
    main()
