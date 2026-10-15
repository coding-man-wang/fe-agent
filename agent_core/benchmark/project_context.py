from pathlib import Path

from agent_core.sdd.model import (
    ProjectContext,
)


class BenchmarkProjectContextError(ValueError):
    pass


class BenchmarkProjectContextProvider:
    """
    v0.1 Benchmark 的项目公开上下文。

    注意：
    这里只提供项目事实，
    不提供 Benchmark 标准答案。

    特别是 relevant_files 必须为空，
    否则会泄漏 expected_files。
    """

    def __init__(
        self,
        *,
        repository_root: str | Path,
    ) -> None:

        self.root = Path(repository_root).resolve(strict=True)

        if not self.root.is_dir():
            raise BenchmarkProjectContextError("repository_root 不是目录")

    def build(
        self,
    ) -> ProjectContext:

        return ProjectContext(
            project_name=("fe-agent-benchmark-web"),
            framework="Vue3",
            language="TypeScript",
            build_tool="Vite",
            package_manager="npm",
            state_management="Pinia",
            test_stack=[
                "Vitest",
                "Vue Test Utils",
                "Playwright",
            ],
            available_components=[],
            engineering_rules=[
                ("优先复用项目现有实现，" "不要为简单需求引入新的框架或依赖"),
                ("修改范围应保持最小化，" "不得无需求扩大改动范围"),
                ("提交前应通过 lint、" "typecheck、unit test 和 build"),
            ],
            directory_summary=[
                "src/api - API 访问层",
                "src/components - 通用与业务组件",
                "src/composables - Vue Composable",
                "src/stores - Pinia 状态管理",
                "src/views - 页面层",
                "src/router - 路由配置",
                "src/types - TypeScript 领域类型",
                "tests/unit - 单元测试",
                "tests/e2e - Playwright E2E 测试",
            ],
            # 非常重要：
            # Benchmark 此阶段不能提前告诉 Agent
            # CASE-001 应该改哪个文件。
            relevant_files=[],
        )
