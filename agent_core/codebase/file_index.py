import os
from collections import Counter
from pathlib import Path

from agent_core.codebase.model import RepositoryFileIndex, FileRecord, FileCategory


class RepositoryFileIndexer:
    DEFAULT_IGNORED_DIRS = {
        ".git",
        ".idea",
        ".vscode",
        "node_modules",
        "dist",
        "build",
        "coverage",
        ".next",
        ".nuxt",
        ".output",
        ".cache",
        ".turbo",
    }

    TEXT_EXTENSIONS = {
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".vue",
        ".json",
        ".html",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".md",
        ".mdx",
        ".yaml",
        ".yml",
        ".toml",
        ".txt",
        ".py",
        ".sh",
    }

    ASSET_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".ico",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".mp3",
        ".mp4",
        ".pdf",
    }

    LANGUAGE_BY_SUFFIX = {
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".vue": "Vue",
        ".py": "Python",
        ".html": "HTML",
        ".css": "CSS",
        ".scss": "SCSS",
        ".sass": "Sass",
        ".less": "Less",
        ".json": "JSON",
        ".md": "Markdown",
        ".mdx": "MDX",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".toml": "TOML",
        ".sh": "Shell",
    }

    SPECIAL_TEXT_FILES = {
        "Dockerfile",
        "Makefile",
        ".gitignore",
        ".npmrc",
        ".nvmrc",
        ".editorconfig",
    }

    CONFIG_FILE_NAMES = {
        "package.json",
        "tsconfig.json",
        "jsconfig.json",
        "vite.config.ts",
        "vite.config.js",
        "webpack.config.js",
        "webpack.config.ts",
        "vitest.config.ts",
        "vitest.config.js",
        "playwright.config.ts",
        "playwright.config.js",
        "eslint.config.js",
        "eslint.config.mjs",
        ".eslintrc",
        ".eslintrc.js",
        ".prettierrc",
        ".prettierrc.json",
    }

    def __init__(
        self,
        *,
        max_indexable_file_size: int = 1_000_000,
        extra_ignored_dirs: set[str] | None = None,
    ):
        self.max_indexable_file_size = max_indexable_file_size

        self.ignored_dirs = set(self.DEFAULT_IGNORED_DIRS)

        if extra_ignored_dirs:
            self.ignored_dirs.update(extra_ignored_dirs)

    def build(self, root_path: str | Path) -> RepositoryFileIndex:
        root = Path(root_path).resolve()

        if not root.exists():
            raise ValueError(f"代码仓库目录不存在: {root}")

        if not root.is_dir():
            raise ValueError(f"代码仓库路径不是目录: {root}")

        records: list[FileRecord] = []

        for current_root, dirs, files in os.walk(
            root,
            topdown=True,
            followlinks=False,
        ):
            dirs[:] = sorted(
                [dirname for dirname in dirs if dirname not in self.ignored_dirs]
            )

            current_path = Path(current_root)

            for filename in files:
                path = current_path / filename

                if path.is_symlink():
                    continue

                record = self._build_record(
                    root=root,
                    path=path,
                )

                records.append(record)

        records.sort(key=lambda item: item.relative_path)

        category_counts = Counter(record.category for record in records)

        return RepositoryFileIndex(
            root_path=str(root),
            total_files=len(records),
            indexable_files=sum(1 for record in records if record.is_indexable),
            category_counts=dict(category_counts),
            files=records,
        )

    def _build_record(self, root: Path, path: Path) -> FileRecord:
        try:
            size_bytes = path.stat().st_size
        except OSError as error:
            raise ValueError(f"无法读取文件信息: {path}") from error

        relative_path = path.relative_to(root).as_posix()

        suffix = path.suffix.lower()

        category = self._detect_category(
            relative_path=relative_path,
            filename=path.name,
            suffix=suffix,
        )

        language = self.LANGUAGE_BY_SUFFIX.get(suffix)

        is_text_file = (
            suffix in self.TEXT_EXTENSIONS or path.name in self.SPECIAL_TEXT_FILES
        )

        if not is_text_file:
            return FileRecord(
                relative_path=relative_path,
                name=path.name,
                suffix=suffix,
                size_bytes=size_bytes,
                category=category,
                language=language,
                is_indexable=False,
                skip_reason=("unsupported_file_type"),
            )

        if size_bytes > self.max_indexable_file_size:
            return FileRecord(
                relative_path=relative_path,
                name=path.name,
                suffix=suffix,
                size_bytes=size_bytes,
                category=category,
                language=language,
                is_indexable=False,
                skip_reason="file_too_large",
            )

        return FileRecord(
            relative_path=relative_path,
            name=path.name,
            suffix=suffix,
            size_bytes=size_bytes,
            category=category,
            language=language,
            is_indexable=True,
            skip_reason=None,
        )

    def _detect_category(
        self,
        *,
        relative_path: str,
        filename: str,
        suffix: str,
    ) -> FileCategory:
        lower_path = relative_path.lower()

        lower_name = filename.lower()

        if (
            "/test/" in f"/{lower_path}/"
            or "/tests/" in f"/{lower_path}/"
            or "/__tests__/" in f"/{lower_path}/"
            or ".test." in lower_name
            or ".spec." in lower_name
        ):
            return "test"

        if (
            filename in self.CONFIG_FILE_NAMES
            or lower_name.startswith("tsconfig")
            or lower_name.startswith("vite.config.")
        ):
            return "config"

        if lower_name.startswith("readme") or suffix in {
            ".md",
            ".mdx",
        }:
            return "documentation"

        if suffix in self.ASSET_EXTENSIONS:
            return "asset"

        if suffix in {
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".vue",
            ".py",
            ".html",
            ".css",
            ".scss",
            ".sass",
            ".less",
        }:
            return "source"

        return "other"
