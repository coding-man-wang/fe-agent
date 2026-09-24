import json
from pathlib import Path

from agent_core.codebase.model import (
    RepositoryProfile,
)
from agent_core.codebase.scanner import (
    RepositoryScanner,
)


def write_package_json(
    root: Path,
    data: dict,
) -> None:
    path = root / "package.json"

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_scan_vue_project(
    tmp_path: Path,
):
    write_package_json(
        tmp_path,
        {
            "name": "admin-web",
            "packageManager": "pnpm@10.0.0",
            "scripts": {
                "dev": "vite",
                "build": "vite build",
                "test": "vitest",
                "test:e2e": "playwright test",
            },
            "dependencies": {
                "vue": "^3.5.0",
                "pinia": "^3.0.0",
            },
            "devDependencies": {
                "typescript": "^5.0.0",
                "vite": "^7.0.0",
                "vitest": "^3.0.0",
                "@playwright/test": "^1.0.0",
            },
        },
    )

    (tmp_path / "tsconfig.json").write_text(
        "{}",
        encoding="utf-8",
    )

    (tmp_path / "vite.config.ts").write_text(
        "",
        encoding="utf-8",
    )

    (tmp_path / "playwright.config.ts").write_text(
        "",
        encoding="utf-8",
    )

    scanner = RepositoryScanner()

    profile = scanner.scan(tmp_path)
    print(profile)
    assert isinstance(
        profile,
        RepositoryProfile,
    )

    assert profile.project_name == "admin-web"

    assert profile.framework == "Vue3"

    assert profile.language == "TypeScript"

    assert profile.build_tool == "Vite"

    assert profile.package_manager == "pnpm"

    assert profile.state_management == "Pinia"

    assert "Vitest" in profile.test_stack

    assert "Playwright" in profile.test_stack

    assert "vite.config.ts" in profile.config_files


def test_detect_npm_from_lockfile(
    tmp_path: Path,
):
    write_package_json(
        tmp_path,
        {
            "name": "demo",
            "dependencies": {"react": "^19.0.0"},
            "devDependencies": {"vite": "^7.0.0"},
        },
    )

    (tmp_path / "package-lock.json").write_text(
        "{}",
        encoding="utf-8",
    )

    scanner = RepositoryScanner()

    profile = scanner.scan(tmp_path)

    assert profile.package_manager == "npm"

    assert profile.framework == "React"


def test_detect_javascript_project(
    tmp_path: Path,
):
    write_package_json(
        tmp_path,
        {
            "name": "js-project",
            "dependencies": {"react": "^19.0.0"},
            "devDependencies": {"vite": "^7.0.0"},
        },
    )

    scanner = RepositoryScanner()

    profile = scanner.scan(tmp_path)

    assert profile.language == "JavaScript"


import pytest


def test_invalid_package_json(
    tmp_path: Path,
):
    (tmp_path / "package.json").write_text(
        "{ invalid json }",
        encoding="utf-8",
    )

    scanner = RepositoryScanner()

    with pytest.raises(
        ValueError,
        match="package.json 格式错误",
    ):
        scanner.scan(tmp_path)


def test_project_path_not_exists(
    tmp_path: Path,
):
    missing = tmp_path / "not-exists"

    scanner = RepositoryScanner()

    with pytest.raises(
        ValueError,
        match="项目目录不存在",
    ):
        scanner.scan(missing)
