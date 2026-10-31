import pytest
from pydantic import ValidationError

from agent_core.code_intelligence.model import (
    CodeSymbol,
    SourceDocument,
    SymbolExtractionResult,
    TechnologyProfile,
)
from agent_core.code_intelligence.profile import (
    RepositoryTechnologyProfileBuilder,
)
from agent_core.code_intelligence.registry import (
    AmbiguousAdapterError,
    DuplicateAdapterError,
    SymbolExtractorRegistry,
)
from agent_core.codebase.model import (
    RepositoryProfile,
)


class FakeVueSymbolAdapter:

    adapter_id = "fake-vue"

    priority = 100

    def supports(
        self,
        *,
        profile,
        document,
    ) -> bool:

        return profile.has("vue") and document.relative_path.endswith(".vue")

    def extract_symbols(
        self,
        *,
        profile,
        document,
    ):

        return SymbolExtractionResult(
            adapter_id=self.adapter_id,
            relative_path=(document.relative_path),
            symbols=[],
        )


class FakeReactSymbolAdapter:

    adapter_id = "fake-react"

    priority = 100

    def supports(
        self,
        *,
        profile,
        document,
    ) -> bool:

        return profile.has("react") and (document.relative_path.endswith(".tsx"))

    def extract_symbols(
        self,
        *,
        profile,
        document,
    ):

        return SymbolExtractionResult(
            adapter_id=self.adapter_id,
            relative_path=(document.relative_path),
            symbols=[],
        )


class FakeGenericTypeScriptAdapter:

    adapter_id = "fake-typescript"

    priority = 10

    def supports(
        self,
        *,
        profile,
        document,
    ) -> bool:

        return profile.has("typescript") and document.relative_path.endswith(
            (
                ".ts",
                ".tsx",
            )
        )

    def extract_symbols(
        self,
        *,
        profile,
        document,
    ):

        return SymbolExtractionResult(
            adapter_id=self.adapter_id,
            relative_path=(document.relative_path),
            symbols=[],
        )


def test_repository_profile_is_decoupled_into_technology_profile():

    repository = RepositoryProfile(
        project_name="demo",
        root_path="/repo",
        framework="Vue3",
        language="TypeScript",
        build_tool="Vite",
        package_manager="pnpm",
        state_management="Pinia",
        test_stack=[
            "Vitest",
            "Playwright",
        ],
    )

    profile = RepositoryTechnologyProfileBuilder().build(repository)

    assert profile.languages == ["typescript"]

    assert profile.frameworks == ["vue"]

    assert profile.build_tools == ["vite"]

    assert profile.test_frameworks == [
        "vitest",
        "playwright",
    ]

    assert profile.has("pinia")

    assert not profile.has("react")


def test_registry_switches_framework_adapter_without_core_branching():

    registry = SymbolExtractorRegistry()

    registry.register(FakeVueSymbolAdapter())

    registry.register(FakeReactSymbolAdapter())

    vue_profile = TechnologyProfile(
        languages=["typescript"],
        frameworks=["vue"],
    )

    react_profile = TechnologyProfile(
        languages=["typescript"],
        frameworks=["react"],
    )

    vue_document = SourceDocument(
        relative_path=("src/views/" "OrdersView.vue"),
        language="typescript",
        content="",
    )

    react_document = SourceDocument(
        relative_path=("src/pages/" "OrdersPage.tsx"),
        language="typescript",
        content="",
    )

    assert (
        registry.resolve(
            profile=vue_profile,
            document=vue_document,
        ).adapter_id
        == "fake-vue"
    )

    assert (
        registry.resolve(
            profile=react_profile,
            document=react_document,
        ).adapter_id
        == "fake-react"
    )


def test_framework_specific_adapter_wins_over_generic_adapter():

    registry = SymbolExtractorRegistry()

    registry.register(FakeReactSymbolAdapter())

    registry.register(FakeGenericTypeScriptAdapter())

    profile = TechnologyProfile(
        languages=["typescript"],
        frameworks=["react"],
    )

    document = SourceDocument(
        relative_path=("src/App.tsx"),
        language="typescript",
        content="",
    )

    adapter = registry.resolve(
        profile=profile,
        document=document,
    )

    assert adapter.adapter_id == "fake-react"


def test_duplicate_adapter_id_fails_closed():

    registry = SymbolExtractorRegistry()

    registry.register(FakeVueSymbolAdapter())

    with pytest.raises(DuplicateAdapterError):
        registry.register(FakeVueSymbolAdapter())


def test_same_priority_matching_adapters_are_ambiguous():

    class AnotherReactAdapter(FakeReactSymbolAdapter):
        adapter_id = "another-react"

    registry = SymbolExtractorRegistry()

    registry.register(FakeReactSymbolAdapter())

    registry.register(AnotherReactAdapter())

    profile = TechnologyProfile(
        languages=["typescript"],
        frameworks=["react"],
    )

    document = SourceDocument(
        relative_path="src/App.tsx",
        language="typescript",
        content="",
    )

    with pytest.raises(AmbiguousAdapterError):
        registry.resolve(
            profile=profile,
            document=document,
        )


def test_symbol_range_must_be_valid():

    with pytest.raises(ValidationError):
        CodeSymbol(
            symbol_id=("src/api/orders.ts" "::function::listOrders"),
            name="listOrders",
            kind="function",
            relative_path=("src/api/orders.ts"),
            start_line=20,
            end_line=10,
        )
