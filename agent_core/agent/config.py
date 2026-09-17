from pydantic import Field, BaseModel


class AgentConfig(BaseModel):
    max_steps:int = Field(
        default=10,
        ge=1,
        description="Agent 单次运行允许的最大推理轮数"
    )

    retry_times: int = Field(
        default=3,
        ge=0,
        description="LLM 或 Tool 临时失败时的最大重试次数",
    )

    retry_delay: int = Field(
        default=1,
        ge=0,
        description="重试间隔秒数",
    )