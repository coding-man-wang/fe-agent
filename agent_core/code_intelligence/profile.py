from __future__ import annotations

import re

from agent_core.code_intelligence.model import (
    TechnologyProfile,
)
from agent_core.codebase.model import (
    RepositoryProfile,
)

_TECHNOLOGY_ALIASES = {
    "vue3": "vue",
    "vue": "vue",
    "react": "react",
    "next.js": "nextjs",
    "nextjs": "nextjs",
    "nuxt": "nuxt",
    "angular": "angular",
    "typescript": "typescript",
    "javascript": "javascript",
    "vite": "vite",
    "webpack": "webpack",
    "rspack": "rspack",
    "parcel": "parcel",
    "vitest": "vitest",
    "jest": "jest",
    "playwright": "playwright",
    "cypress": "cypress",
    "testing library": "testing-library",
    "pinia": "pinia",
    "vuex": "vuex",
    "redux": "redux",
    "redux toolkit": "redux-toolkit",
    "zustand": "zustand",
    "mobx": "mobx",
    "npm": "npm",
    "pnpm": "pnpm",
    "yarn": "yarn",
    "bun": "bun",
}


def canonicalize_technology_id(
    value: str,
) -> str:
    """
    将 Repository Discovery 输出的
    人类可读名称转换为稳定 Technology ID。

    例如：

        Vue3
        → vue

        Next.js
        → nextjs

        Redux Toolkit
        → redux-toolkit

    具体技术名称只允许集中出现在
    Technology Boundary，
    不允许泄漏到 Symbol Core。
    """

    normalized = value.strip().lower()

    alias = _TECHNOLOGY_ALIASES.get(normalized)

    if alias is not None:
        return alias

    fallback = re.sub(
        r"[^a-z0-9]+",
        "-",
        normalized,
    ).strip("-")

    return fallback


def _optional(
    value: str | None,
) -> list[str]:

    if not value:
        return []

    return [canonicalize_technology_id(value)]


def _many(
    values: list[str],
) -> list[str]:

    result: list[str] = []

    seen: set[str] = set()

    for value in values:

        canonical = canonicalize_technology_id(value)

        if not canonical:
            continue

        if canonical in seen:
            continue

        seen.add(canonical)

        result.append(canonical)

    return result


class RepositoryTechnologyProfileBuilder:
    """
    RepositoryProfile
        ↓
    TechnologyProfile

    这是旧 Repository Discovery
    与新 Code Intelligence 之间
    唯一允许的转换边界之一。
    """

    def build(
        self,
        repository: RepositoryProfile,
    ) -> TechnologyProfile:

        return TechnologyProfile(
            languages=_optional(repository.language),
            frameworks=_optional(repository.framework),
            build_tools=_optional(repository.build_tool),
            test_frameworks=_many(repository.test_stack),
            package_managers=_optional(repository.package_manager),
            state_management=_optional(repository.state_management),
        )
