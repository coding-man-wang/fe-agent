import json

from agent_core.codebase.model import (
    RepositoryProfile,
)
from agent_core.codebase.symbol_retrieval_model import (
    SymbolCandidate,
)
from agent_core.context.model import (
    CodeContextItem,
    CodeContextSegment,
    ContextBudgetUsage,
    TaskContextBundle,
)
from agent_core.context.renderer import (
    ContextRenderer,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TaskDefinition,
    TechnicalSpec,
)


def test_renderer_exposes_symbol_window_metadata():

    content = (
        "import { request } "
        "from './request'\n\n"
        "export function listOrders() {\n"
        "  return request()\n"
        "}"
    )

    symbol = SymbolCandidate(
        symbol_id=("src/api/orders.ts" "::function::listOrders"),
        name="listOrders",
        kind="function",
        relative_path=("src/api/orders.ts"),
        start_line=20,
        end_line=22,
        score=0.9,
        source="evidence",
    )

    item = CodeContextItem(
        relative_path=("src/api/orders.ts"),
        category="source",
        language="TypeScript",
        content_mode="symbol_window",
        content=content,
        start_line=None,
        end_line=None,
        segments=[
            CodeContextSegment(
                start_line=1,
                end_line=1,
                sources=["import_coverage"],
            ),
            CodeContextSegment(
                start_line=20,
                end_line=22,
                sources=["symbol"],
                symbol_ids=[symbol.symbol_id],
            ),
        ],
        symbol_candidates=[symbol],
        covered_symbol_ids=[symbol.symbol_id],
        content_chars=len(content),
        source_file_chars=500,
    )

    bundle = TaskContextBundle(
        requirement=(
            RequirementSpec(
                title="orders",
                summary="orders",
            )
        ),
        technical_spec=(
            TechnicalSpec(
                summary="orders",
            )
        ),
        task=TaskDefinition(
            task_id="TASK-001",
            task_type="code_change",
            title="orders",
            objective="orders",
            source_step_order=1,
        ),
        repository=RepositoryProfile(
            project_name="demo",
            root_path="/repo",
        ),
        code_context=[item],
        budget=ContextBudgetUsage(
            max_files=5,
            max_total_chars=10_000,
            max_chars_per_file=5_000,
            used_files=1,
            used_chars=len(content),
            symbol_window_files=["src/api/orders.ts"],
            symbol_window_saved_chars=(500 - len(content)),
        ),
    )

    rendered = ContextRenderer().render(bundle)

    payload = json.loads(rendered.user_message)

    code = payload["code_context"][0]

    assert code["content_mode"] == "symbol_window"

    assert code["segments"][1]["start_line"] == 20

    assert code["coverage"]["covered_symbol_ids"] == [symbol.symbol_id]

    assert code["coverage"]["missing_symbol_ids"] == []

    assert "不连续" in rendered.system_message

    assert code["content"] == content
