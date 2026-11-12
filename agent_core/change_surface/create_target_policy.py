from pathlib import Path, PurePosixPath


class CreateTargetPolicyError(ValueError):
    pass


class CreateTargetPolicy:
    """
    v1 Prospective Write Target Policy。

    当前只开放：

        test Task
            ↓
        tests/unit/*.test.ts
        tests/e2e/*.spec.ts

    不允许：

        - 创建生产代码
        - 创建新目录
        - 路径穿越
        - 通过 symlink 写出仓库
        - 覆盖已经存在的文件
    """

    TEST_ROOTS = (
        "tests/unit",
        "tests/e2e",
    )

    @staticmethod
    def canonical_component_test_targets(
        source_paths: set[str],
    ) -> dict[str, str]:
        """Repository-evidence-based unit targets, never Benchmark Oracle paths.

        For directly located Vue views/components only. Ambiguous basenames are
        excluded rather than granting a potentially wrong write target.
        """
        targets: dict[str, str] = {}
        ambiguous: set[str] = set()
        for source in sorted(source_paths):
            path = PurePosixPath(source)
            if (
                path.parent.as_posix() not in {"src/views", "src/components"}
                or path.suffix != ".vue"
            ):
                continue
            key = path.stem.casefold()
            target = f"tests/unit/{path.stem}.test.ts"
            if key in targets:
                ambiguous.add(key)
            else:
                targets[key] = target
        return {key: path for key, path in targets.items() if key not in ambiguous}

    @staticmethod
    def canonicalize_component_test_create_path(
        relative_path: str,
        targets: dict[str, str],
    ) -> str:
        """Canonicalize only identifiable Vue test aliases; do not widen scope.

        e.g. usersView.test.ts / UsersView.pagination.test.ts -> UsersView.test.ts.
        A file unrelated to an evidenced component remains subject to the
        existing CreateTargetPolicy checks (no guessed write permission).
        """
        path = PurePosixPath(relative_path)
        if path.parent.as_posix() != "tests/unit" or not path.name.endswith(".test.ts"):
            return relative_path

        base = path.name[: -len(".test.ts")].casefold()
        matches = {
            target
            for stem, target in targets.items()
            if base == stem or base.startswith(stem + ".")
        }
        return next(iter(matches)) if len(matches) == 1 else relative_path

    @classmethod
    def discover_test_roots(
        cls,
        repository_root: str | Path,
    ) -> list[str]:
        root = Path(repository_root).resolve(strict=True)

        if not root.is_dir():
            raise CreateTargetPolicyError("repository_root 不是目录")

        result: list[str] = []

        for relative_root in cls.TEST_ROOTS:
            candidate = root / relative_root

            if not candidate.exists() or candidate.is_symlink():
                continue

            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                continue

            if resolved.is_dir() and resolved.is_relative_to(root):
                result.append(relative_root)

        return result

    @classmethod
    def validate_test_create_path(
        cls,
        *,
        repository_root: str | Path,
        relative_path: str,
        allowed_roots: list[str] | None = None,
    ) -> str:
        root = Path(repository_root).resolve(strict=True)

        if (
            not relative_path
            or "\\" in relative_path
            or "\x00" in relative_path
            or ":" in relative_path
        ):
            raise CreateTargetPolicyError(f"非法 create path: " f"{relative_path}")

        relative = PurePosixPath(relative_path)

        if relative.is_absolute():
            raise CreateTargetPolicyError("create target " "不能是绝对路径")

        if any(part in {"", ".", ".."} for part in relative.parts):
            raise CreateTargetPolicyError("create target " "包含非法路径段")

        normalized = relative.as_posix()

        roots = (
            allowed_roots
            if allowed_roots is not None
            else cls.discover_test_roots(root)
        )

        parent = relative.parent.as_posix()

        # v1 不允许顺手创建目录。
        if parent not in roots:
            raise CreateTargetPolicyError(
                "v1 只能直接在已有测试目录" "中创建测试文件: " f"{normalized}"
            )

        if parent == "tests/unit":
            if not normalized.endswith(".test.ts"):
                raise CreateTargetPolicyError(
                    "unit create target " "必须以 .test.ts 结尾"
                )

        elif parent == "tests/e2e":
            if not normalized.endswith(".spec.ts"):
                raise CreateTargetPolicyError(
                    "e2e create target " "必须以 .spec.ts 结尾"
                )

        else:
            raise CreateTargetPolicyError("当前不是允许的测试目录")

        target = root / normalized

        # broken symlink:
        # exists() 可能 False，
        # is_symlink() 仍可能 True。
        if target.exists() or target.is_symlink():
            raise CreateTargetPolicyError(
                "create target " "当前已经存在: " f"{normalized}"
            )

        parent_path = target.parent

        if parent_path.is_symlink():
            raise CreateTargetPolicyError("create target " "父目录不能是 symlink")

        try:
            parent_resolved = parent_path.resolve(strict=True)
        except OSError as exc:
            raise CreateTargetPolicyError("create target " "父目录不存在") from exc

        if not parent_resolved.is_relative_to(root):
            raise CreateTargetPolicyError("create target " "超出仓库范围")

        if not parent_resolved.is_dir():
            raise CreateTargetPolicyError("create target " "父路径不是目录")

        return normalized
