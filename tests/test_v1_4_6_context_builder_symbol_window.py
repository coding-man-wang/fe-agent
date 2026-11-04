from pathlib import Path

import pytest

from agent_core.codebase.model import (
    CodeChunk,
    HybridRelevantFile,
    HybridRelevantFileSelection,
    RepositoryProfile,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)
from agent_core.context.builder import (
    ProjectContextBuilder,
    SymbolWindowCoverageError,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


def requirement():
    return RequirementSpec(
        title="test",
        summary="test",
    )


def technical_spec():
    return TechnicalSpec(
        summary="test",
    )


def task():
    return TaskDefinition(
        task_id="TASK-001",
        task_type="code_change",
        title="test",
        objective="test",
        source_step_order=1,
    )


def repository(root: Path):
    return RepositoryProfile(
        project_name="test",
        root_path=str(root),
    )


def candidate(
    *,
    symbol_id,
    path,
    name,
    kind,
    start,
    end,
):
    return SymbolCandidate(
        symbol_id=symbol_id,
        name=name,
        kind=kind,
        relative_path=path,
        start_line=start,
        end_line=end,
        score=0.9,
        source="evidence",
    )


def relevant_file(
    *,
    path,
    content,
    rank,
    symbols=None,
    chunk=None,
    role="reference",
):
    return HybridRelevantFile(
        relative_path=path,
        category="source",
        language="TypeScript",
        size_bytes=len(content.encode("utf-8")),
        retrieval_score=1.0,
        retrieval_rank=rank,
        policy_score=0,
        selection_score=1.0,
        context_role=role,
        symbol_candidates=(symbols or []),
        best_semantic_chunk=chunk,
        best_semantic_score=(0.9 if chunk is not None else None),
    )


def selection(files):
    return HybridRelevantFileSelection(
        candidate_count=len(files),
        eligible_count=len(files),
        selected_count=len(files),
        total_selected_bytes=sum(item.size_bytes for item in files),
        selected_files=files,
    )


def build_bundle(
    *,
    root,
    files,
    max_chars_per_file=8000,
    max_total_chars=30000,
):
    return ProjectContextBuilder().build(
        repository_root=root,
        requirement=requirement(),
        technical_spec=technical_spec(),
        task=task(),
        repository=repository(root),
        selection=selection(files),
        max_files=8,
        max_total_chars=(max_total_chars),
        max_chars_per_file=(max_chars_per_file),
    )


def test_reference_file_uses_symbol_window(
    tmp_path: Path,
):
    path = tmp_path / "src" / "api" / "demo.ts"

    path.parent.mkdir(parents=True)

    lines = [
        "import { request } " "from './request'",
        "import type { Result } " "from './types'",
        *[f"const filler_{line} = {line}" for line in range(3, 61)],
        "export function " "target(): Result {",
        "  return request<Result>()",
        "}",
        *[f"const filler_{line} = {line}" for line in range(64, 121)],
    ]

    content = "\n".join(lines)

    path.write_text(
        content,
        encoding="utf-8",
    )

    symbol_id = "src/api/demo.ts" "::function::target"

    file = relevant_file(
        path="src/api/demo.ts",
        content=content,
        rank=1,
        symbols=[
            candidate(
                symbol_id=(symbol_id),
                path=("src/api/demo.ts"),
                name="target",
                kind="function",
                start=61,
                end=63,
            )
        ],
    )

    bundle = build_bundle(
        root=tmp_path,
        files=[file],
        max_chars_per_file=800,
    )

    item = bundle.code_context[0]

    assert item.content_mode == "symbol_window"

    assert symbol_id in item.covered_symbol_ids

    assert "export function target()" in item.content

    assert "filler_100" not in item.content

    assert bundle.budget.symbol_window_files == ["src/api/demo.ts"]

    assert bundle.budget.symbol_window_saved_chars > 0


def test_small_implementation_target_stays_full_file(
    tmp_path: Path,
):
    path = tmp_path / "src" / "api" / "demo.ts"

    path.parent.mkdir(parents=True)

    content = "export function " "target() { return 1 }\n"

    path.write_text(
        content,
        encoding="utf-8",
    )

    symbol_id = "src/api/demo.ts" "::function::target"

    file = relevant_file(
        path="src/api/demo.ts",
        content=content,
        rank=1,
        role="implementation_target",
        symbols=[
            candidate(
                symbol_id=(symbol_id),
                path=("src/api/demo.ts"),
                name="target",
                kind="function",
                start=1,
                end=1,
            )
        ],
    )

    bundle = build_bundle(
        root=tmp_path,
        files=[file],
        max_chars_per_file=1000,
    )

    item = bundle.code_context[0]

    assert item.content_mode == "full_file"

    assert symbol_id in item.covered_symbol_ids


def test_symbol_mode_compresses_unmapped_baseline_file(
    tmp_path: Path,
):
    src = tmp_path / "src"

    src.mkdir()

    anchor_content = "export function " "anchor() { return 1 }\n"

    noise_lines = [f"const value_{line} = {line}" for line in range(1, 31)]

    noise_content = "\n".join(noise_lines)

    (src / "anchor.ts").write_text(
        anchor_content,
        encoding="utf-8",
    )

    (src / "noise.ts").write_text(
        noise_content,
        encoding="utf-8",
    )

    anchor_id = "src/anchor.ts" "::function::anchor"

    anchor = relevant_file(
        path="src/anchor.ts",
        content=anchor_content,
        rank=1,
        symbols=[
            candidate(
                symbol_id=(anchor_id),
                path=("src/anchor.ts"),
                name="anchor",
                kind="function",
                start=1,
                end=1,
            )
        ],
    )

    chunk = CodeChunk(
        chunk_id=("src/noise.ts:10-12"),
        relative_path=("src/noise.ts"),
        start_line=10,
        end_line=12,
        content="\n".join(noise_lines[9:12]),
        language="TypeScript",
        category="source",
    )

    noise = relevant_file(
        path="src/noise.ts",
        content=noise_content,
        rank=2,
        chunk=chunk,
    )

    bundle = build_bundle(
        root=tmp_path,
        files=[
            anchor,
            noise,
        ],
    )

    noise_item = next(
        item for item in bundle.code_context if (item.relative_path == "src/noise.ts")
    )

    assert noise_item.content_mode == "semantic_chunk"

    assert noise_item.start_line == 10

    assert noise_item.end_line == 12

    assert "value_10" in noise_item.content


def test_legacy_no_symbol_selection_keeps_full_file(
    tmp_path: Path,
):
    path = tmp_path / "src" / "demo.ts"

    path.parent.mkdir(parents=True)

    content = "export const value = 1\n"

    path.write_text(
        content,
        encoding="utf-8",
    )

    chunk = CodeChunk(
        chunk_id=("src/demo.ts:1-1"),
        relative_path=("src/demo.ts"),
        start_line=1,
        end_line=1,
        content=(content.strip()),
        language="TypeScript",
        category="source",
    )

    file = relevant_file(
        path="src/demo.ts",
        content=content,
        rank=1,
        chunk=chunk,
    )

    bundle = build_bundle(
        root=tmp_path,
        files=[file],
        max_chars_per_file=1000,
    )

    assert bundle.code_context[0].content_mode == "full_file"


def test_semantic_char_clipping_updates_end_line(
    tmp_path: Path,
):
    src = tmp_path / "src"

    src.mkdir()

    anchor_content = "const x = 1\n"

    (src / "anchor.ts").write_text(
        anchor_content,
        encoding="utf-8",
    )

    big_lines = [
        "const a = 1",
        "const b = 2",
        "const c = 3",
        "const d = 4",
    ]

    big_content = "\n".join(big_lines)

    (src / "big.ts").write_text(
        big_content,
        encoding="utf-8",
    )

    anchor = relevant_file(
        path="src/anchor.ts",
        content=anchor_content,
        rank=1,
        symbols=[
            candidate(
                symbol_id=("src/anchor.ts" "::variable::x"),
                path=("src/anchor.ts"),
                name="x",
                kind="variable",
                start=1,
                end=1,
            )
        ],
    )

    chunk = CodeChunk(
        chunk_id=("src/big.ts:1-4"),
        relative_path=("src/big.ts"),
        start_line=1,
        end_line=4,
        content=big_content,
        language="TypeScript",
        category="source",
    )

    big = relevant_file(
        path="src/big.ts",
        content=big_content,
        rank=2,
        chunk=chunk,
    )

    bundle = build_bundle(
        root=tmp_path,
        files=[
            anchor,
            big,
        ],
        max_chars_per_file=20,
        max_total_chars=1000,
    )

    item = next(
        item for item in bundle.code_context if (item.relative_path == "src/big.ts")
    )

    assert item.content_mode == "semantic_chunk"

    assert item.end_line == 1


def test_builder_fails_closed_when_symbol_cannot_fit(
    tmp_path: Path,
):
    path = tmp_path / "src" / "huge.ts"

    path.parent.mkdir(parents=True)

    content = "\n".join(
        [
            "export function huge() {",
            ("  const a = '" + ("x" * 200) + "'"),
            "  return a",
            "}",
        ]
    )

    path.write_text(
        content,
        encoding="utf-8",
    )

    file = relevant_file(
        path="src/huge.ts",
        content=content,
        rank=1,
        symbols=[
            candidate(
                symbol_id=("src/huge.ts" "::function::huge"),
                path=("src/huge.ts"),
                name="huge",
                kind="function",
                start=1,
                end=4,
            )
        ],
    )

    with pytest.raises(SymbolWindowCoverageError):
        build_bundle(
            root=tmp_path,
            files=[file],
            max_chars_per_file=80,
        )
