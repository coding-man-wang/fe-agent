from pathlib import Path

import pytest

from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)
from agent_core.codebase.model import (
    RepositoryFileIndex,
)


def test_build_file_index(
    tmp_path: Path,
):
    src = tmp_path / "src"
    src.mkdir()

    api = src / "api"
    api.mkdir()

    tests = tmp_path / "tests"
    tests.mkdir()

    public = tmp_path / "public"
    public.mkdir()

    (src / "App.vue").write_text(
        "<template></template>",
        encoding="utf-8",
    )

    (api / "user.ts").write_text(
        "export const getUsers = () => {}",
        encoding="utf-8",
    )

    (tests / "user.spec.ts").write_text(
        "describe('user', () => {})",
        encoding="utf-8",
    )

    (tmp_path / "package.json").write_text(
        "{}",
        encoding="utf-8",
    )

    (tmp_path / "README.md").write_text(
        "# Demo",
        encoding="utf-8",
    )

    (public / "logo.png").write_bytes(b"fake image")

    indexer = RepositoryFileIndexer()

    index = indexer.build(tmp_path)

    assert isinstance(
        index,
        RepositoryFileIndex,
    )

    assert index.total_files == 6

    assert index.indexable_files == 5

    file_map = {file.relative_path: file for file in index.files}

    assert file_map["src/App.vue"].category == "source"

    assert file_map["tests/user.spec.ts"].category == "test"

    assert file_map["package.json"].category == "config"

    assert file_map["README.md"].category == "documentation"

    assert file_map["public/logo.png"].is_indexable is False


def test_ignore_generated_directories(
    tmp_path: Path,
):
    src = tmp_path / "src"
    src.mkdir()

    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()

    dist = tmp_path / "dist"
    dist.mkdir()

    git = tmp_path / ".git"
    git.mkdir()

    (src / "main.ts").write_text(
        "console.log('hello')",
        encoding="utf-8",
    )

    (node_modules / "vue.js").write_text(
        "fake",
        encoding="utf-8",
    )

    (dist / "bundle.js").write_text(
        "fake",
        encoding="utf-8",
    )

    (git / "config").write_text(
        "fake",
        encoding="utf-8",
    )

    indexer = RepositoryFileIndexer()

    index = indexer.build(tmp_path)

    paths = {item.relative_path for item in index.files}

    assert paths == {"src/main.ts"}


def test_large_file_not_indexable(
    tmp_path: Path,
):
    large_file = tmp_path / "large.ts"

    large_file.write_text(
        "a" * 200,
        encoding="utf-8",
    )

    indexer = RepositoryFileIndexer(max_indexable_file_size=100)

    index = indexer.build(tmp_path)

    record = index.files[0]

    assert record.is_indexable is False

    assert record.skip_reason == "file_too_large"


def test_binary_asset_not_indexable(
    tmp_path: Path,
):
    image = tmp_path / "logo.png"

    image.write_bytes(b"\x89PNG")

    indexer = RepositoryFileIndexer()

    index = indexer.build(tmp_path)

    record = index.files[0]

    assert record.category == "asset"

    assert record.is_indexable is False

    assert record.skip_reason == "unsupported_file_type"


def test_root_not_exists(
    tmp_path: Path,
):
    missing = tmp_path / "missing"

    indexer = RepositoryFileIndexer()

    with pytest.raises(
        ValueError,
        match="代码仓库目录不存在",
    ):
        indexer.build(missing)


def test_extra_ignored_directory(
    tmp_path: Path,
):
    src = tmp_path / "src"
    src.mkdir()

    generated = tmp_path / "generated"
    generated.mkdir()

    (src / "main.ts").write_text(
        "",
        encoding="utf-8",
    )

    (generated / "api.ts").write_text(
        "",
        encoding="utf-8",
    )

    indexer = RepositoryFileIndexer(extra_ignored_dirs={"generated"})

    index = indexer.build(tmp_path)

    paths = [file.relative_path for file in index.files]

    assert paths == ["src/main.ts"]
