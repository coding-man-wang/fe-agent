from openai import AsyncOpenAI

from agent_core.codebase.embedding import (
    EmbeddingProvider,
)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    使用 OpenAI-compatible Embedding API
    把文本转换成向量。

    注意：

    Embedding Provider 和 LLM Provider
    是两个独立配置。

    不要默认认为：
    能调用 Chat 的 Provider
    就一定支持 Embedding。
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        batch_size: int = 64,
        request_timeout_seconds: float = 60.0,
    ) -> None:

        if not model.strip():
            raise ValueError("Embedding model 不能为空")

        if not api_key.strip():
            raise ValueError("Embedding api_key 不能为空")

        if batch_size < 1:
            raise ValueError("batch_size 必须 >= 1")

        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds " "必须大于 0")

        self.model = model

        self.batch_size = batch_size

        client_options = {
            "api_key": api_key,
            "timeout": (request_timeout_seconds),
            # Embedding 是独立外部调用，
            # v0.1 先直接使用 SDK Retry。
            "max_retries": 2,
        }

        if base_url:
            client_options["base_url"] = base_url

        self.client = AsyncOpenAI(**client_options)

    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        for text in texts:

            if not text.strip():
                raise ValueError("Embedding 输入不能为空")

        all_vectors: list[list[float]] = []

        # -------------------------------
        # 分批 Embedding。
        #
        # 避免仓库变大后一次请求
        # 放太多代码 Chunk。
        # -------------------------------

        for start in range(
            0,
            len(texts),
            self.batch_size,
        ):

            batch = texts[start : start + self.batch_size]

            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float",
            )

            # 根据 index 排序，
            # 确保结果与输入顺序一致。
            data = sorted(
                response.data,
                key=lambda item: item.index,
            )

            if len(data) != len(batch):

                raise ValueError("Embedding 返回数量" "与输入数量不一致")

            for item in data:

                all_vectors.append(list(item.embedding))

        if len(all_vectors) != len(texts):

            raise ValueError("Embedding 最终数量异常")

        dimensions = {len(vector) for vector in all_vectors}

        if len(dimensions) > 1:

            raise ValueError("Embedding 向量维度不一致")

        return all_vectors
