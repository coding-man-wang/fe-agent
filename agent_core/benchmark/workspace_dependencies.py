from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class WorkspaceDependencyError(RuntimeError):
    pass


class NodeModulesWorkspaceProvisioner:
    """Reuse a prepared benchmark repo ``node_modules`` inside Git worktrees.

    Full execution intentionally does not run ``npm install`` inside every task
    worktree.  The source dependency tree therefore becomes benchmark runtime
    infrastructure and must be validated *before* it is reused.

    The previous implementation only checked whether ``node_modules`` existed.
    That is insufficient on npm projects because a production-only/incomplete
    install can exist while dev binaries such as eslint, vue-tsc, vitest and vite
    are missing from ``node_modules/.bin``.
    """

    DEFAULT_REQUIRED_BINS: tuple[str, ...] = (
        "eslint",
        "vue-tsc",
        "vitest",
        "vite",
    )

    def __init__(
        self,
        *,
        dependency_source_root: str | Path,
        required_bins: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.source_root = Path(dependency_source_root).resolve(strict=True)
        self.source_node_modules = self.source_root / "node_modules"
        self.required_bins = tuple(required_bins or self.DEFAULT_REQUIRED_BINS)

    def require_source_ready(self) -> None:
        """Fail fast when the reusable dependency tree is not benchmark-ready."""

        if shutil.which("node") is None:
            raise WorkspaceDependencyError(
                "找不到 node 可执行文件。请先安装 Node.js 并确认 node 在 PATH 中。"
            )

        if shutil.which("npm") is None:
            raise WorkspaceDependencyError(
                "找不到 npm 可执行文件。请先安装 npm 并确认 npm 在 PATH 中。"
            )

        self._require_node_modules_ready(
            node_modules=self.source_node_modules,
            label="Benchmark 源仓库",
        )

    def provision(self, workspace_root: str | Path) -> None:
        """Attach validated dependencies to a worktree and validate the target."""

        # Validate the source first.  Do not create a worktree link to a broken
        # dependency tree and then discover the problem during Verification.
        self.require_source_ready()

        workspace = Path(workspace_root).resolve(strict=True)
        target = workspace / "node_modules"

        if target.exists():
            self._require_node_modules_ready(
                node_modules=target,
                label="Execution Worktree",
            )
            return

        try:
            os.symlink(
                self.source_node_modules,
                target,
                target_is_directory=True,
            )
        except OSError:
            if os.name != "nt":
                raise WorkspaceDependencyError(
                    "无法为 Worktree 创建 node_modules symlink"
                )

            result = subprocess.run(
                [
                    "cmd",
                    "/c",
                    "mklink",
                    "/J",
                    str(target),
                    str(self.source_node_modules),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 or not target.exists():
                raise WorkspaceDependencyError(
                    "无法为 Worktree 创建 node_modules junction: "
                    + (result.stderr or result.stdout)
                )

        # The link/junction existing is not enough; npm scripts depend on .bin.
        self._require_node_modules_ready(
            node_modules=target,
            label="Execution Worktree",
        )

    def missing_bins(self, node_modules: str | Path | None = None) -> list[str]:
        root = (
            Path(node_modules)
            if node_modules is not None
            else self.source_node_modules
        )
        return [
            name
            for name in self.required_bins
            if not self._has_npm_bin(root, name)
        ]

    @staticmethod
    def _has_npm_bin(node_modules: Path, name: str) -> bool:
        bin_root = node_modules / ".bin"
        # npm creates .cmd launchers on Windows and extension-less shims on
        # POSIX.  Accept both so validation remains portable and testable.
        candidates = (
            bin_root / name,
            bin_root / f"{name}.cmd",
            bin_root / f"{name}.exe",
            bin_root / f"{name}.ps1",
        )
        return any(path.is_file() or path.is_symlink() for path in candidates)

    def _require_node_modules_ready(
        self,
        *,
        node_modules: Path,
        label: str,
    ) -> None:
        if not node_modules.is_dir():
            raise WorkspaceDependencyError(
                f"{label}缺少 node_modules。请在 {self.source_root} 执行 "
                "`npm ci` 后再运行 Full Execution Benchmark。"
            )

        bin_root = node_modules / ".bin"
        if not bin_root.is_dir():
            raise WorkspaceDependencyError(
                f"{label}的 node_modules/.bin 不存在。当前依赖安装不完整。"
                f"请在 {self.source_root} 删除/重建依赖并执行 `npm ci`。"
            )

        missing = self.missing_bins(node_modules)
        if not missing:
            return

        raise WorkspaceDependencyError(
            f"{label}缺少 Verification 所需 npm binary: {', '.join(missing)}。"
            "这些工具属于 devDependencies；请确认没有使用 --omit=dev / "
            "production-only 安装。建议在 "
            f"{self.source_root} 执行 `npm ci`，然后重试 Benchmark。"
        )
