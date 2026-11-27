"""Read a bounded, explicitly authorized committed Git tree; never live source.

The temporary mirror is for the existing local retrieval algorithms. It is not
an execution workspace and is never passed to the patch writer.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
import unicodedata
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from time import monotonic
from types import MappingProxyType
from typing import Mapping

from pydantic import Field, model_validator

from .contracts import Contract, RetrievalBudget, PathText

_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SUFFIXES = frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue",
                       ".html", ".css", ".scss", ".less", ".sass", ".json",
                       ".md", ".mdx", ".py", ".yaml", ".yml", ".toml", ".txt"})
_DENIED = frozenset({".git", ".ssh", ".fe-agent", ".aws", ".azure", "node_modules",
                     "dist", "build", "coverage", ".next", ".nuxt", ".output",
                     ".venv", "secrets", "credentials", "oracle", "oracles", "benchmark",
                     "benchmarks", "__pycache__", ".idea", ".vscode"})
_DEVICE = re.compile(r"(?:con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\..*)?\Z", re.I)
_SECRET_NAME = re.compile(r"(?:^|[._-])(?:secrets?|credentials?|passwords?|private[_-]?key|api[_-]?key|oracle|expected[_-]?files)(?:$|[._-])", re.I)


class CatalogError(ValueError):
    pass


def safe_path(path: str) -> bool:
    if (not path or len(path) > 300 or path.startswith("/")
            or any(c in path for c in '\\:<>"|?*')
            or any(ord(c) < 32 or ord(c) == 127 for c in path)):
        return False
    parts = path.split("/")
    for part in parts:
        lower = part.casefold()
        if (part in {"", ".", ".."} or part.endswith((" ", "."))
                or _DEVICE.fullmatch(part) or lower in _DENIED
                or lower.startswith((".env", ".git"))
                or lower in {".npmrc", ".pypirc", ".netrc", "id_rsa", "id_ed25519"}
                or _SECRET_NAME.search(lower)):
            return False
    return True


class ReadPolicy(Contract):
    allowed_prefixes: tuple[PathText, ...] = Field(default=(), max_length=50)
    allowed_paths: tuple[PathText, ...] = Field(default=(), max_length=1000)

    @model_validator(mode="after")
    def validate_policy(self):
        if not self.allowed_prefixes and not self.allowed_paths:
            raise ValueError("explicit read authorization is required")
        for path in (*self.allowed_prefixes, *self.allowed_paths):
            if not safe_path(path):
                raise ValueError("invalid or denied authorized path/prefix")
        for path in self.allowed_paths:
            if PurePosixPath(path).suffix.lower() not in _SUFFIXES:
                raise ValueError("unsupported explicitly authorized file type")
        return self

    def permits(self, path: str) -> bool:
        return (safe_path(path) and PurePosixPath(path).suffix.lower() in _SUFFIXES
                and (path in self.allowed_paths or any(
                    path.startswith(prefix + "/") for prefix in self.allowed_prefixes)))


def _git(root: Path, args: list[str], *, timeout: float, input_bytes=None) -> bytes:
    # The environment is caller-owned; ignore overrides that could redirect Git
    # to a different repository, worktree or object directory.
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env.update(GIT_NO_REPLACE_OBJECTS="1", GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    try:
        completed = subprocess.run(["git", "--no-replace-objects", "-C", str(root),
                                    "--literal-pathspecs", *args], input=input_bytes,
                                   capture_output=True, check=True, timeout=timeout, env=env)
        return completed.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise CatalogError(f"git command failed: {args[0]}") from exc


def head_revision(root: str | Path) -> str:
    root = Path(root).resolve(strict=True)
    revision = _git(root, ["rev-parse", "--verify", "HEAD^{commit}"], timeout=10).decode().strip()
    if not _SHA.fullmatch(revision):
        raise CatalogError("invalid commit revision")
    return revision


def assert_clean_revision(root: str | Path, revision: str):
    root = Path(root).resolve(strict=True)
    if head_revision(root) != revision:
        raise CatalogError("workspace revision mismatch")
    if _git(root, ["status", "--porcelain=v1", "-z", "--untracked-files=normal"], timeout=10):
        raise CatalogError("workspace is not clean")


class CommittedCatalog:
    def __init__(self, revision: str, files: Mapping[str, str], *, excluded_count: int = 0):
        if not _SHA.fullmatch(revision):
            raise CatalogError("invalid revision")
        self._revision = revision
        self._files = MappingProxyType(dict(files))
        self._excluded_count = excluded_count

    @property
    def revision(self):
        return self._revision

    @property
    def files(self):
        return self._files

    @property
    def excluded_count(self):
        return self._excluded_count

    def digest(self, path: str) -> str:
        return hashlib.sha256(self.files[path].encode("utf-8")).hexdigest()

    @classmethod
    def from_git(cls, root: str | Path, policy: ReadPolicy, budget: RetrievalBudget,
                 *, revision: str | None = None):
        root = Path(root).resolve(strict=True)
        deadline = monotonic() + budget.setup_timeout_seconds

        def run(*args, input_bytes=None):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise CatalogError("catalog setup timeout")
            return _git(root, list(args), timeout=remaining, input_bytes=input_bytes)

        actual_root = Path(run("rev-parse", "--show-toplevel").decode().strip()).resolve()
        if actual_root != root:
            raise CatalogError("repository root must be the Git top-level directory")
        if revision is None:
            revision = run("rev-parse", "--verify", "HEAD^{commit}").decode().strip()
        if not _SHA.fullmatch(revision):
            raise CatalogError("revision must be a full commit SHA")
        if run("rev-parse", "--verify", revision + "^{commit}").decode().strip() != revision:
            raise CatalogError("revision is not a commit")

        tree = run("ls-tree", "-r", "-l", "-z", "--full-tree", revision)
        if len(tree) > budget.max_tree_bytes:
            raise CatalogError("tree metadata budget exceeded")
        entries: list[tuple[str, str, int]] = []
        total = 0
        excluded = 0
        aliases: dict[str, str] = {}
        file_keys: set[str] = set()
        for entry in tree.split(b"\0"):
            if not entry:
                continue
            meta, raw_path = entry.split(b"\t", 1)
            try:
                path = raw_path.decode("utf-8")
            except UnicodeDecodeError:
                excluded += 1
                continue
            if not policy.permits(path):
                excluded += 1
                continue
            mode, kind, oid, raw_size = meta.decode("ascii").split()
            if kind != "blob" or mode not in {"100644", "100755"} or not _SHA.fullmatch(oid):
                excluded += 1
                continue
            size = int(raw_size)
            if size > budget.max_file_bytes:
                excluded += 1
                continue
            # Disallow collisions on case-insensitive and Unicode-normalizing
            # filesystems, including parent directory/file collisions.
            parts = path.split("/")
            for i in range(1, len(parts) + 1):
                prefix = "/".join(parts[:i])
                key = unicodedata.normalize("NFC", prefix).casefold()
                prior = aliases.setdefault(key, prefix)
                if prior != prefix or (i < len(parts) and key in file_keys):
                    raise CatalogError("case/Unicode path collision")
                if i == len(parts):
                    file_keys.add(key)
            entries.append((path, oid, size))
            total += size
            if len(entries) > budget.max_catalog_files or total > budget.max_catalog_bytes:
                raise CatalogError("authorized catalog exceeds file/byte budget; narrow scope")
        if not entries:
            raise CatalogError("no authorized regular text files")

        # One bounded batch avoids spawning a Git process per file on Windows.
        # Each output header and byte count is checked before use.
        blob_data = run("cat-file", "--batch", input_bytes=("\n".join(e[1] for e in entries) + "\n").encode("ascii"))
        offset = 0
        files: dict[str, str] = {}
        for path, oid, size in entries:
            end = blob_data.find(b"\n", offset)
            if end < 0:
                raise CatalogError("truncated blob header")
            expected = f"{oid} blob {size}".encode("ascii")
            if blob_data[offset:end] != expected:
                raise CatalogError("blob identity mismatch")
            start = end + 1
            data = blob_data[start:start + size]
            offset = start + size + 1
            if len(data) != size or blob_data[offset - 1:offset] != b"\n":
                raise CatalogError("truncated blob")
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                excluded += 1
                continue
            if b"\0" in data or text.startswith("version https://git-lfs.github.com/spec/v1\n"):
                excluded += 1
                continue
            files[path] = text
        if offset != len(blob_data):
            raise CatalogError("unexpected blob data")
        if not set(policy.allowed_paths) <= files.keys():
            raise CatalogError("an explicitly authorized file is unavailable or unsafe")
        if not files:
            raise CatalogError("no supported text blobs")
        return cls(revision, files, excluded_count=excluded)

    @contextmanager
    def mirror(self):
        with tempfile.TemporaryDirectory(prefix="fe-agent-v16-") as temp:
            root = Path(temp)
            for relative_path, content in self.files.items():
                if not safe_path(relative_path):
                    raise CatalogError("unsafe mirror path")
                path = root.joinpath(*relative_path.split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation prevents overwriting a colliding path.
                with path.open("xb") as stream:
                    stream.write(content.encode("utf-8"))
            yield root
