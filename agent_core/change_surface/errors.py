class ChangeSurfaceError(Exception):
    """
    Change Surface 子系统基础异常。
    """

    pass


class InvestigationArtifactBuildError(ChangeSurfaceError):
    """
    InvestigationArtifact 构建失败。

    典型情况：

    - 当前 Task 不是 investigation
    - 没有 Repository Code Context
    - LLM 返回非法 JSON
    - LLM 引用了未提供的 Repository 文件
    - 返回的 task_id 与当前 Task 不一致
    """

    pass
