import json
from pathlib import Path
from typing import Any

from agent_core.codebase.model import RepositoryProfile


class RepositoryScanner:
    def scan(self, root_path) -> RepositoryProfile:
        root = Path(root_path).resolve()

        if not root.exists():
            raise ValueError(f"项目目录不存在: {root}")

        if not root.is_dir():
            raise ValueError(f"项目路径不是目录: {root}")

        package_json_path = root / "package.json"
        print(111, package_json_path)
        package_data: dict[
            str,
            Any,
        ] = {}

        if package_json_path.exists():
            package_data = self._read_package_json(package_json_path)

        dependencies = self._collect_dependencies(package_data)
        print(package_data)
        print(dependencies)
        return RepositoryProfile(
            project_name=package_data.get("name") or root.name,
            root_path=str(root),
            framework=self._detect_framework(dependencies),
            language=self._detect_language(
                root=root,
                dependencies=dependencies,
            ),
            build_tool=self._detect_build_tool(
                root=root,
                dependencies=dependencies,
            ),
            package_manager=self._detect_package_manager(
                root=root,
                package_data=package_data,
            ),
            state_management=(self._detect_state_management(dependencies)),
            test_stack=self._detect_test_stack(dependencies),
            scripts=dict(package_data.get("scripts", {})),
            dependencies=sorted(dependencies),
            config_files=(self._detect_config_files(root)),
        )

    def _read_package_json(
        self,
        path: Path,
    ) -> dict[str, Any]:

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
                print(222, data)

        except json.JSONDecodeError as error:
            raise ValueError(f"package.json 格式错误: {path}") from error
        except Exception as error:
            print(error)

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError("package.json 根节点必须是对象")
        return data

    def _collect_dependencies(
        self,
        package_data: dict[str, Any],
    ) -> set[str]:

        result: set[str] = set()

        for field_name in [
            "dependencies",
            "devDependencies",
            "peerDependencies",
        ]:

            values = package_data.get(field_name, {})

            if isinstance(
                values,
                dict,
            ):
                result.update(values.keys())

        return result

    def _detect_framework(
        self,
        dependencies: set[str],
    ) -> str:

        if "nuxt" in dependencies:
            return "Nuxt"

        if "next" in dependencies:
            return "Next.js"

        if "@angular/core" in dependencies:
            return "Angular"

        if "vue" in dependencies:
            return "Vue3"

        if "react" in dependencies:
            return "React"

        return "Unknown"

    def _detect_language(
        self,
        root: Path,
        dependencies: set[str],
    ) -> str:

        if (root / "tsconfig.json").exists() or "typescript" in dependencies:
            return "TypeScript"

        return "JavaScript"

    def _detect_build_tool(
        self,
        root: Path,
        dependencies: set[str],
    ) -> str | None:

        if "vite" in dependencies:
            return "Vite"

        if "webpack" in dependencies:
            return "Webpack"

        if "@rspack/core" in dependencies:
            return "Rspack"

        if "parcel" in dependencies:
            return "Parcel"

        if "next" in dependencies:
            return "Next.js"

        if "nuxt" in dependencies:
            return "Nuxt"

        return None

    def _detect_package_manager(
        self,
        root: Path,
        package_data: dict[str, Any],
    ) -> str | None:

        package_manager = package_data.get("packageManager")

        if isinstance(
            package_manager,
            str,
        ):
            return package_manager.split(
                "@",
                1,
            )[0]

        if (root / "pnpm-lock.yaml").exists():
            return "pnpm"

        if (root / "yarn.lock").exists():
            return "yarn"

        if (root / "package-lock.json").exists():
            return "npm"

        if (root / "bun.lock").exists():
            return "bun"

        if (root / "bun.lockb").exists():
            return "bun"

        return None

    def _detect_state_management(
        self,
        dependencies: set[str],
    ) -> str | None:

        if "pinia" in dependencies:
            return "Pinia"

        if "@reduxjs/toolkit" in dependencies:
            return "Redux Toolkit"

        if "redux" in dependencies:
            return "Redux"

        if "zustand" in dependencies:
            return "Zustand"

        if "mobx" in dependencies:
            return "MobX"

        if "vuex" in dependencies:
            return "Vuex"

        return None

    def _detect_test_stack(
        self,
        dependencies: set[str],
    ) -> list[str]:

        result: list[str] = []

        candidates = [
            (
                "vitest",
                "Vitest",
            ),
            (
                "jest",
                "Jest",
            ),
            (
                "@playwright/test",
                "Playwright",
            ),
            (
                "cypress",
                "Cypress",
            ),
            (
                "@testing-library/react",
                "Testing Library",
            ),
            (
                "@testing-library/vue",
                "Testing Library",
            ),
        ]

        for dependency, name in candidates:

            if dependency in dependencies and name not in result:
                result.append(name)

        return result

    def _detect_config_files(
        self,
        root: Path,
    ) -> list[str]:

        candidates = [
            "package.json",
            "tsconfig.json",
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
        ]

        result: list[str] = []

        for filename in candidates:
            if (root / filename).exists():
                result.append(filename)

        return result
