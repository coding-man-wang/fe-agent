from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from apps.agent_cli.benchmark_symbol_window_v1_4_6 import (
    run_real_benchmark,
)


def rank_of(report: dict, path: str):
    if report["status"] != "complete":
        return None

    for item in report["ranked_files"]:
        if item["relative_path"] == path:
            return {
                "rank": item["rank"],
                "score": item["score"],
            }

    return None


async def main() -> None:
    load_dotenv()

    result = await run_real_benchmark(
        include_noise_shadow=True,
        include_rerank_shadow=True,
    )

    rerank = result["rerank_shadow"]
    order_task = rerank["order_task"]
    user_task = rerank["user_control_task"]

    # 本阶段只检查诊断是否完整，
    # 不把排名质量直接作为通过门槛。
    completed = order_task["status"] == "complete" and user_task["status"] == "complete"

    user_filters = "src/components/UserFilters.vue"

    output = {
        "version": "v1.4.8-P2-A",
        "mode": "rerank_shadow",
        "diagnostic_status": ("complete" if completed else "incomplete"),
        "repository_root": result["repository_root"],
        "selection_changed": rerank["selection_changed"],
        "candidate_count": len(rerank["candidate_paths"]),
        "order_task": order_task,
        "user_control_task": user_task,
        "user_filters_comparison": {
            "order_task": rank_of(
                order_task,
                user_filters,
            ),
            "user_task": rank_of(
                user_task,
                user_filters,
            ),
        },
        "p1_integrity": {
            "symbol_status": (result["noise_shadow"]["symbol_evaluation"]["status"]),
            "file_recall": (result["noise_shadow"]["file_quality"]["recall"]),
        },
    }

    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
    )

    if not completed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
