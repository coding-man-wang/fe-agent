from abc import (
    ABC,
    abstractmethod,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class RerankDocument(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
    )

    document_id: str

    text: str


class RerankHit(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
    )

    document_id: str

    rank: int = Field(
        ge=1,
    )

    score: float


class Reranker(ABC):

    @abstractmethod
    async def rerank(
        self,
        *,
        query: str,
        documents: list[RerankDocument],
        top_n: int,
    ) -> list[RerankHit]: ...
