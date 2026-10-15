import re
from pathlib import Path

from agent_core.benchmark.suite_model import (
    BenchmarkSuiteResult,
)


class BenchmarkSuiteStoreError(ValueError):
    pass


class BenchmarkSuiteStore:

    def __init__(
        self,
        runtime_root: str | Path,
    ) -> None:

        self.runtime_root = Path(runtime_root).resolve(strict=True)

        if not self.runtime_root.is_dir():
            raise BenchmarkSuiteStoreError("runtime_root 不是目录")

        self.directory = self.runtime_root / ".fe-agent" / "benchmark" / "suites"

    def _path(
        self,
        suite_id: str,
    ) -> Path:

        if not re.fullmatch(
            r"[0-9a-f]{32}",
            suite_id,
        ):
            raise BenchmarkSuiteStoreError("非法 suite_id")

        return self.directory / f"{suite_id}.json"

    def save(
        self,
        suite: BenchmarkSuiteResult,
    ) -> None:

        path = self._path(suite.suite_id)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_suffix(".json.tmp")

        temporary.write_text(
            suite.model_dump_json(
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary.replace(path)

    def load(
        self,
        suite_id: str,
    ) -> BenchmarkSuiteResult:

        path = self._path(suite_id)

        if not path.is_file():
            raise BenchmarkSuiteStoreError("Benchmark Suite 不存在")

        try:

            suite = BenchmarkSuiteResult.model_validate_json(
                path.read_text(encoding="utf-8")
            )

        except Exception as exc:

            raise BenchmarkSuiteStoreError("Benchmark Suite 文件非法") from exc

        if suite.suite_id != suite_id:
            raise BenchmarkSuiteStoreError("Suite ID 不一致")

        return suite
