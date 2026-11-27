import hashlib
import re
import subprocess
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator, ValidationError
from agent_core.tools.base import Tool
from agent_core.tools.registry import ToolRegistry
from .model import Observation


class Snapshot:
    """Read an explicitly approved subset of a committed Git tree, not live files."""

    def __init__(self, revision: str, files: Mapping[str, str]):
        self.revision = revision
        self.files = MappingProxyType(dict(files))

    @classmethod
    def from_git(
        cls,
        root: str | Path,
        approved_paths: list[str],
        *,
        max_file_bytes: int = 64000,
        max_total_bytes: int = 256000,
    ):
        root = Path(root).resolve(strict=True)
        if not approved_paths or len(approved_paths) > 100:
            raise ValueError("approve 1–100 explicit repository paths")
        if max_file_bytes <= 0 or max_total_bytes <= 0:
            raise ValueError("invalid snapshot budget")

        def git(*args):
            return subprocess.run(
                ["git", "-C", str(root), "--literal-pathspecs", *args],
                check=True,
                capture_output=True,
                timeout=10,
            ).stdout

        revision = git("rev-parse", "--verify", "HEAD^{commit}").decode().strip()
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
            raise ValueError("invalid revision")
        files = {}
        total = 0
        for path in dict.fromkeys(approved_paths):
            # Reject paths likely to contain credentials even when supplied accidentally.
            p = PurePosixPath(path)
            if (
                not path
                or p.is_absolute()
                or p.as_posix() != path
                or "\\" in path
                or ":" in path
                or any(part in (".", "..", ".git") for part in p.parts)
                or any(
                    part.lower().startswith(".env")
                    or part.lower()
                    in {"node_modules", "secrets", "credentials", ".ssh"}
                    for part in p.parts
                )
                or p.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}
            ):
                raise ValueError(f"unsafe approved path: {path}")
            entries = git("ls-tree", "-z", revision, "--", path).split(b"\0")
            entries = [e for e in entries if e]
            if len(entries) != 1:
                raise ValueError(f"not a single tracked file: {path}")
            meta, actual_path = entries[0].split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            if (
                actual_path.decode() != path
                or kind != "blob"
                or mode not in {"100644", "100755"}
            ):
                raise ValueError(f"not a regular tracked file: {path}")
            size = int(git("cat-file", "-s", oid))
            if size > max_file_bytes or total + size > max_total_bytes:
                raise ValueError(f"snapshot byte budget exceeded: {path}")
            data = git("cat-file", "blob", oid)
            if len(data) != size or b"\0" in data:
                raise ValueError(f"not a supported text blob: {path}")
            files[path] = data.decode("utf-8")
            total += size
        return cls(revision, files)


class ListArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prefix: str = Field(default="", max_length=240)


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=300)
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=80, ge=1)

    @model_validator(mode="after")
    def validate_range(self):
        if self.end_line < self.start_line or self.end_line - self.start_line >= 100:
            raise ValueError("read at most 100 lines")
        return self


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=15, ge=1, le=30)


class SnapshotTool(Tool):
    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot


class ListSnapshotFiles(SnapshotTool):
    name = "list_files"
    description = "List approved committed text paths; optional literal prefix."
    args_schema = ListArgs

    async def execute(self, arguments):
        args = self.args_schema.model_validate(arguments)
        return {
            "revision": self.snapshot.revision,
            "paths": sorted(
                p for p in self.snapshot.files if p.startswith(args.prefix)
            ),
        }


class InvestigationValidationError(ValueError):

    def __init__(
        self,
        code: str,
        message: str,
    ) -> None:

        self.code = code

        super().__init__(message)


class ReadSnapshotFile(SnapshotTool):

    name = "read_file"

    description = (
        "Read exact numbered lines from an approved "
        "committed path. At most 100 lines. "
        "Use numbered_lines for citation line numbers "
        "and exact quotes."
    )

    args_schema = ReadArgs

    async def execute(self, arguments):

        args = self.args_schema.model_validate(arguments)

        if args.path not in self.snapshot.files:

            return {
                "success": False,
            }

        lines = self.snapshot.files[args.path].splitlines()

        if args.start_line > len(lines):

            return {
                "success": False,
            }

        selected = lines[args.start_line - 1 : args.end_line]

        return {
            "path": args.path,
            "revision": self.snapshot.revision,
            "start_line": args.start_line,
            "end_line": (args.start_line + len(selected) - 1),
            # 保留已有字段，保证兼容性。
            "lines": selected,
            # 新增：显式行号，供模型引用。
            "numbered_lines": [
                {
                    "line": line_number,
                    "content": content,
                }
                for line_number, content in enumerate(
                    selected,
                    start=args.start_line,
                )
            ],
            "sha256": hashlib.sha256(
                self.snapshot.files[args.path].encode()
            ).hexdigest(),
        }


class SearchSnapshotCode(SnapshotTool):
    name = "search_code"
    description = "Find literal case-insensitive text in the approved snapshot. Read matches for evidence."
    args_schema = SearchArgs

    async def execute(self, arguments):
        args = self.args_schema.model_validate(arguments)
        hits = []
        for path, text in sorted(self.snapshot.files.items()):
            for i, line in enumerate(text.splitlines(), 1):
                if args.query.casefold() in line.casefold():
                    if len(hits) == args.limit:
                        return {"hits": hits, "truncated": True}
                    hits.append({"path": path, "line": i, "snippet": line[:240]})
        return {"hits": hits, "truncated": False}


def repository_registry(snapshot: Snapshot) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        ListSnapshotFiles(snapshot),
        ReadSnapshotFile(snapshot),
        SearchSnapshotCode(snapshot),
    ):
        registry.register(tool)
    return registry


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=1, max_length=2000)
    call_id: str = Field(min_length=1)
    path: str
    line: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=2000)


class InvestigationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: list[Finding] = Field(max_length=30)
    unresolved_questions: list[str] = Field(max_length=30)


def normalize_investigation_json(
    content: str,
) -> str:
    """
    兼容完整 Markdown JSON 代码块。

    仅处理外层格式，不修复 JSON 内容，
    不删除字段，不改变证据。
    """

    normalized = content.strip()

    match = re.fullmatch(
        r"```(?:json)?[ \t]*\r?\n" r"([\s\S]*?)\r?\n```",
        normalized,
        flags=re.IGNORECASE,
    )

    if match:
        normalized = match.group(1).strip()

    return normalized


def validate_investigation(
    content: str,
    observations: list[Observation],
) -> None:

    # ==================================
    # 1. Normalize Output
    # ==================================

    normalized = normalize_investigation_json(content)

    # ==================================
    # 2. Schema Validation
    # ==================================

    try:

        report = InvestigationReport.model_validate_json(normalized)

    except ValidationError as exc:

        raise InvestigationValidationError(
            code="invalid_report_schema",
            message="调查报告不符合 JSON Schema",
        ) from exc

    # ==================================
    # 3. Findings Validation
    # ==================================

    if not report.findings:

        raise InvestigationValidationError(
            code="missing_findings",
            message="调查报告没有有效结论",
        )

    # ==================================
    # 4. Build Evidence Index
    # ==================================

    reads = {
        observation.call_id: observation.data
        for observation in observations
        if (observation.success and observation.tool_name == "read_file")
    }

    # ==================================
    # 5. Citation Validation
    # ==================================

    for finding in report.findings:

        read = reads.get(finding.call_id)

        if (
            not read
            or read["path"] != finding.path
            or not (read["start_line"] <= finding.line <= read["end_line"])
        ):

            raise InvestigationValidationError(
                code="invalid_citation",
                message=(f"无效证据引用: " f"{finding.path}:{finding.line}"),
            )

        actual_line = read["lines"][finding.line - read["start_line"]]

        if not finding.quote.strip() or finding.quote not in actual_line:

            raise InvestigationValidationError(
                code="citation_text_mismatch",
                message=(f"证据内容不匹配: " f"{finding.path}:{finding.line}"),
            )

    # ==================================
    # 6. Evidence Completeness Gate
    # ==================================

    if report.unresolved_questions:

        raise InvestigationValidationError(
            code="unresolved_questions",
            message=(
                "调查存在未解决问题，数量: " f"{len(report.unresolved_questions)}"
            ),
        )
