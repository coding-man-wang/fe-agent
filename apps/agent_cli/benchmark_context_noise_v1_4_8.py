from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from apps.agent_cli.benchmark_symbol_window_v1_4_6 import (
    run_real_benchmark,
)

BASELINE_CONTEXT_CHARS = 3902
BASELINE_NOISE_CHARS = 1321


async def main() -> None:
    load_dotenv()

    # 只运行一次 Retrieval。
    # A/B 与 Shadow 共用相同的 Selection。
    result = await run_real_benchmark(
        include_ab=True,
        include_noise_shadow=True,
    )

    if "noise_shadow" not in result:
        raise RuntimeError(
            "run_real_benchmark 未返回 noise_shadow，"
            "请检查 include_noise_shadow 接入"
        )

    shadow = result["noise_shadow"]
    symbol = shadow["symbol_evaluation"]
    file_quality = shadow["file_quality"]

    # Completion Gate
    gates = {
        "required_file_recall": (file_quality["recall"] == 1.0),
        "required_symbol_coverage": (symbol["symbol_coverage"] == 1.0),
        "budget_integrity": (symbol["budget_preserved"] is True),
        "source_integrity": (symbol["source_integrity_preserved"] is True),
        "metadata_integrity": (symbol["metadata_integrity_preserved"] is True),
        "noise_reduced": (file_quality["irrelevant_chars"] < BASELINE_NOISE_CHARS),
        "context_not_increased": (shadow["context_chars"] <= BASELINE_CONTEXT_CHARS),
    }

    overall_status = "passed" if all(gates.values()) else "failed"

    output = {
        "version": "v1.4.8-P1",
        "mode": "shadow",
        "overall_status": overall_status,
        "repository_root": result["repository_root"],
        "baseline": {
            "context_chars": BASELINE_CONTEXT_CHARS,
            "noise_chars": BASELINE_NOISE_CHARS,
        },
        "selection_diagnostics": result.get("selection_diagnostics"),
        "shadow": shadow,
        "completion_gates": gates,
    }

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
