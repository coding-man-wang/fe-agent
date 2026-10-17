from typing import Any

from openai import (
    AsyncOpenAI,
)

from agent_core.codebase.reranker import (
    RerankDocument,
    RerankHit,
    Reranker,
)


class OpenAICompatibleReranker(Reranker):

    DEFAULT_INSTRUCT = (
        "Given a software change request, "
        "rank candidate code files by how "
        "necessary they are to understand "
        "or implement the requested change. "
        "Prefer directly affected "
        "implementation files and relevant "
        "tests. Down-rank files that only "
        "share similar domain terminology."
    )

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        request_timeout_seconds: float = 60,
        instruct: str | None = None,
    ) -> None:

        if not model.strip():
            raise ValueError("rerank model 不能为空")

        if not api_key.strip():
            raise ValueError("rerank api_key 不能为空")

        self.model = model

        self.instruct = instruct or self.DEFAULT_INSTRUCT

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout_seconds,
            max_retries=2,
        )

    async def rerank(
        self,
        *,
        query: str,
        documents: list[RerankDocument],
        top_n: int,
    ) -> list[RerankHit]:

        if not query.strip():
            raise ValueError("rerank query 不能为空")

        if not documents:
            return []

        if top_n < 1:
            raise ValueError("top_n 必须 >= 1")

        document_texts = [document.text for document in documents]
        try:
            response = await self.client.post(
                "/reranks",
                body={
                    "model": self.model,
                    "query": query,
                    "documents": (document_texts),
                    "top_n": min(
                        top_n,
                        len(documents),
                    ),
                    "instruct": (self.instruct),
                },
                cast_to=object,
            )
        except Exception as exc:
            raise RuntimeError(
                "Rerank request failed: "
                f"model={self.model}, "
                f"documents={len(documents)}"
            ) from exc

        payload = self._to_dict(response)

        raw_results = payload.get("results")

        # 兼容部分 Provider
        # output.results 的返回形式。
        if raw_results is None:

            output = payload.get(
                "output",
                {},
            )

            raw_results = output.get(
                "results",
                [],
            )

        hits: list[RerankHit] = []

        for rank, item in enumerate(
            raw_results,
            start=1,
        ):

            index = int(item["index"])

            if index < 0 or index >= len(documents):
                raise ValueError("Rerank 返回了非法 " "document index")

            hits.append(
                RerankHit(
                    document_id=(documents[index].document_id),
                    rank=rank,
                    score=float(item["relevance_score"]),
                )
            )

        return hits

    @staticmethod
    def _to_dict(
        value: Any,
    ) -> dict:

        if isinstance(
            value,
            dict,
        ):
            return value

        if hasattr(
            value,
            "model_dump",
        ):
            return value.model_dump()

        if hasattr(
            value,
            "__dict__",
        ):
            return dict(value.__dict__)

        raise TypeError("无法解析 Rerank Response")
