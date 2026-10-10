from typing import Protocol

from agent_core.coding.model import (
    ReadOnlyCodingResult,
)
from agent_core.context.model import (
    RenderedContext,
    TaskContextBundle,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


class ContextFactory(Protocol):
    """
    根据 TaskDefinition 构建完整 Task Context。

    真实实现：
        DefaultTaskContextFactory
    """

    async def build(
        self,
        task: TaskDefinition,
    ) -> TaskContextBundle: ...


class CodingAnalyzer(Protocol):
    """
    普通只读 Coding Analysis 能力。
    """

    async def analyze(
        self,
        bundle: TaskContextBundle,
    ) -> ReadOnlyCodingResult: ...


class PreparedCodingAnalyzer(Protocol):
    """
    支持“消息准备”和“使用已准备消息分析”的 Analyzer。

    Durable Workflow 使用这两个方法保证：

        被 Hash 的消息
        ==
        真正发送给模型的消息
    """

    def prepare_messages(
        self,
        bundle: TaskContextBundle,
    ) -> RenderedContext: ...

    async def analyze_prepared(
        self,
        *,
        bundle: TaskContextBundle,
        messages: RenderedContext,
    ) -> ReadOnlyCodingResult: ...


class DurableCodingAnalyzer(
    CodingAnalyzer,
    PreparedCodingAnalyzer,
    Protocol,
):
    """
    Durable Workflow 所需要的完整 Analyzer 能力。

    真实实现：
        ReadOnlyCodingAgent
    """

    pass
