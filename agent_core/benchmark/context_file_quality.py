from pydantic import BaseModel, ConfigDict, Field

from agent_core.context.model import TaskContextBundle


class ContextFileOracle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_files: list[str] = Field(default_factory=list)
    useful_files: list[str] = Field(default_factory=list)
    irrelevant_files: list[str] = Field(default_factory=list)


class ContextFileQualityResult(BaseModel):
    recall: float
    precision: float

    matched_required: list[str]
    missing_required: list[str]
    irrelevant_hits: list[str]

    irrelevant_chars: int
    irrelevant_ratio: float


class ContextFileQualityEvaluator:

    def evaluate(
        self,
        *,
        bundle: TaskContextBundle,
        oracle: ContextFileOracle,
    ) -> ContextFileQualityResult:

        required = set(oracle.required_files)
        useful = set(oracle.useful_files)
        irrelevant = set(oracle.irrelevant_files)

        if not required:
            raise ValueError("File Oracle 必须定义 Required Files")

        if (required & useful) or (required & irrelevant) or (useful & irrelevant):
            raise ValueError("File Oracle 角色存在交集")

        selected = {item.relative_path for item in bundle.code_context}

        matched = required & selected
        missing = required - selected

        irrelevant_hits = selected & irrelevant

        total_chars = sum(len(item.content) for item in bundle.code_context)

        irrelevant_chars = sum(
            len(item.content)
            for item in bundle.code_context
            if item.relative_path in irrelevant
        )

        # 未标注文件不自动归为无关。
        # 但它们也不能计为 Precision 的有效命中。
        relevant_hits = selected & (required | useful)

        return ContextFileQualityResult(
            recall=len(matched) / len(required),
            precision=(len(relevant_hits) / len(selected) if selected else 0.0),
            matched_required=sorted(matched),
            missing_required=sorted(missing),
            irrelevant_hits=sorted(irrelevant_hits),
            irrelevant_chars=irrelevant_chars,
            irrelevant_ratio=(irrelevant_chars / total_chars if total_chars else 0.0),
        )
