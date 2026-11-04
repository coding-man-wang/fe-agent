from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from apps.agent_cli.benchmark_symbol_window_v1_4_6 import (
    run_real_benchmark,
)


async def main() -> None:
    load_dotenv()

    real = await run_real_benchmark(
        include_ab=True,
    )

    comparison = real["context_ab"]
    previous_evaluation = real["evaluation"]

    overall_status = (
        "passed"
        if (
            previous_evaluation["status"] == "passed"
            and comparison["status"] == "passed"
        )
        else "failed"
    )

    output = {
        "version": "v1.4.7",
        "overall_status": overall_status,
        "case_id": "REAL-ORDER-CONTEXT-AB",
        "repository_root": real["repository_root"],
        "symbol_pipeline": real["symbol_pipeline"],
        "context_ab": comparison,
        "v1_4_6_integrity": {
            "status": previous_evaluation["status"],
            "symbol_coverage": (previous_evaluation["symbol_coverage"]),
            "source_integrity_preserved": (
                previous_evaluation["source_integrity_preserved"]
            ),
            "metadata_integrity_preserved": (
                previous_evaluation["metadata_integrity_preserved"]
            ),
            "budget_preserved": (previous_evaluation["budget_preserved"]),
        },
    }

    print("=" * 80)
    print("v1.4.7 Context Quality A/B Benchmark")
    print("=" * 80)

    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
    )

    if overall_status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
