import re
from pathlib import Path

from agent_core.benchmark.result_model import (
    BenchmarkCaseResult,
)


class BenchmarkResultStoreError(ValueError):
    pass


class BenchmarkResultStore:

    def __init__(
        self,
        runtime_root: str | Path,
    ) -> None:

        self.runtime_root = Path(runtime_root).resolve(strict=True)

        if not self.runtime_root.is_dir():
            raise BenchmarkResultStoreError("runtime_root 不是目录")

        self.directory = self.runtime_root / ".fe-agent" / "benchmark" / "results"

    def _path(
        self,
        run_id: str,
    ) -> Path:

        if not re.fullmatch(
            r"[0-9a-f]{32}",
            run_id,
        ):
            raise BenchmarkResultStoreError("非法 run_id")

        return self.directory / f"{run_id}.json"

    def save(
        self,
        result: BenchmarkCaseResult,
    ) -> None:

        path = self._path(result.run_id)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_suffix(".json.tmp")

        temporary.write_text(
            result.model_dump_json(
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary.replace(path)

    def load(
        self,
        run_id: str,
    ) -> BenchmarkCaseResult:

        path = self._path(run_id)

        if not path.is_file():
            raise BenchmarkResultStoreError("Benchmark Result 不存在")

        try:

            result = BenchmarkCaseResult.model_validate_json(
                path.read_text(encoding="utf-8")
            )

        except Exception as exc:

            raise BenchmarkResultStoreError("Benchmark Result 文件非法") from exc

        if result.run_id != run_id:
            raise BenchmarkResultStoreError("Run ID 不一致")

        return result
