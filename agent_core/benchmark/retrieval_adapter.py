from agent_core.codebase.model import (
    SemanticSearchResponse,
)


class EmptySemanticCodeSearcher:
    """
    Benchmark Retrieval v0.1 使用的
    Semantic Search Adapter。

    当前不执行真正的 Semantic Retrieval，
    永远返回空结果。

    目的：

    继续复用现有 TaskRetrievalService 的
    Hybrid Pipeline：

        Lexical
        +
        Semantic
        +
        Hybrid Fusion
        +
        File Selection

    但当前阶段只测 deterministic
    Lexical Retrieval baseline。

    后续接入真正 Semantic Search 后，
    可以直接比较：

        lexical baseline
        vs
        hybrid semantic
    """

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> SemanticSearchResponse:

        return SemanticSearchResponse(
            query=query,
            results=[],
        )
