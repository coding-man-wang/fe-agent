import json
from pathlib import Path

from pydantic import ValidationError

from agent_core.benchmark.case_model import (
    BenchmarkCase,
)


class BenchmarkCaseLoadError(ValueError):
    pass


class BenchmarkCaseLoader:

    def __init__(
        self,
        case_root: str | Path,
    ) -> None:

        self.case_root = Path(case_root).resolve(strict=True)

        self.runnable_root = self.case_root / "runnable"

        if not self.case_root.is_dir():
            raise BenchmarkCaseLoadError("case_root 不是目录")

    def load(
        self,
        case_id: str,
    ) -> BenchmarkCase:

        self._validate_case_id(case_id)

        path = self.runnable_root / f"{case_id}.json"

        if not path.is_file():
            raise BenchmarkCaseLoadError("Benchmark Case 不存在: " f"{case_id}")

        try:

            raw = path.read_text(encoding="utf-8")

            data = json.loads(raw)

            case = BenchmarkCase.model_validate(data)

        except (
            OSError,
            json.JSONDecodeError,
            ValidationError,
        ) as exc:

            raise BenchmarkCaseLoadError("Benchmark Case 非法: " f"{case_id}") from exc

        if case.case_id != case_id:
            raise BenchmarkCaseLoadError("Case 文件名与 case_id " "不一致")

        return case

    def list_case_ids(
        self,
    ) -> list[str]:
        """
        返回 case_root 下所有合法命名的
        Benchmark Case ID。

        这里只负责发现文件名，
        不代表 Case 内容一定合法。

        真正执行时仍然通过 load()
        完成 Schema Validation。
        """

        case_ids: list[str] = []

        for path in self.runnable_root.glob("CASE-*.json"):

            if not path.is_file():
                continue

            case_id = path.stem

            try:
                self._validate_case_id(case_id)
            except BenchmarkCaseLoadError:
                continue

            case_ids.append(case_id)

        return sorted(case_ids)

    @staticmethod
    def _validate_case_id(
        case_id: str,
    ) -> None:

        if not case_id or "/" in case_id or "\\" in case_id or ".." in case_id:
            raise BenchmarkCaseLoadError("非法 case_id")

        if not case_id.startswith("CASE-"):
            raise BenchmarkCaseLoadError("case_id 必须以 CASE- 开头")
