import json

from pydantic import ValidationError

from agent_core.change_surface.errors import (
    InvestigationArtifactBuildError,
)
from agent_core.change_surface.model import (
    InvestigationArtifact,
    ObservedRepositoryFile,
)
from agent_core.context.model import (
    TaskContextBundle,
)
from agent_core.llm.base import LLM
from agent_core.llm.model import (
    ChatRequest,
    Message,
)


class InvestigationArtifactBuilder:
    """
    Investigation Artifact Builder v0.1

    作用：

        Repository Retrieval
                ↓
        TaskContextBundle
                ↓
        真实代码分析
                ↓
        InvestigationArtifact

    它负责回答：

        “这次 Investigation
         从代码仓中确认了哪些事实？”

    注意：

    它不是 Change Surface Resolver。

    Builder 负责 Observation：

        我看到了什么？

    Resolver 负责 Decision：

        最终哪些文件要改？
    """

    def __init__(
        self,
        *,
        llm: LLM,
    ) -> None:
        self.llm = llm

    async def build(
        self,
        *,
        bundle: TaskContextBundle,
    ) -> InvestigationArtifact:

        self._validate_input(
            bundle=bundle,
        )

        schema = InvestigationArtifact.model_json_schema()

        system_prompt = self._build_system_prompt(
            schema=schema,
        )

        payload = self._build_payload(
            bundle=bundle,
        )

        request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content=system_prompt,
                ),
                Message(
                    role="user",
                    content=json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
            ],
        )

        response = await self.llm.chat(request)

        content = response.content.strip()

        if not content:
            raise (
                InvestigationArtifactBuildError(
                    "LLM 返回了空的 " "InvestigationArtifact"
                )
            )

        try:
            artifact = InvestigationArtifact.model_validate_json(content)

        except (
            ValidationError,
            ValueError,
        ) as exc:
            raise (
                InvestigationArtifactBuildError(
                    "LLM 返回的 " "InvestigationArtifact " "不满足 Schema"
                )
            ) from exc

        artifact = self._canonicalize_paths(
            artifact=artifact,
            bundle=bundle,
        )

        self._validate_output(
            artifact=artifact,
            bundle=bundle,
        )

        return artifact

    @staticmethod
    def _validate_input(
        *,
        bundle: TaskContextBundle,
    ) -> None:

        task = bundle.task

        if task.task_type != "investigation":
            raise (
                InvestigationArtifactBuildError(
                    "InvestigationArtifactBuilder "
                    "只能处理 investigation Task，"
                    f"当前 task_type="
                    f"{task.task_type}"
                )
            )

        if not bundle.code_context:
            raise (
                InvestigationArtifactBuildError(
                    "Investigation Task "
                    "没有 Repository Code Context，"
                    "无法生成可靠的 "
                    "InvestigationArtifact"
                )
            )

    @staticmethod
    def _build_system_prompt(
        *,
        schema: dict,
    ) -> str:

        return f"""
你是一名资深软件仓库分析工程师。

你的任务不是修改代码，而是分析当前
Investigation Task 提供的 Repository Code Context，
生成结构化 InvestigationArtifact。

你必须严格区分：

Observation：
    从实际代码中确认的事实。

Decision：
    后续最终决定修改哪些文件。

你当前只负责 Observation，
不要代替 Change Surface Resolver
做最终修改范围决策。


==============================
核心规则
==============================

1. 只能分析输入中 code_context 提供的文件。

2. observed_files 中的 relative_path
   必须来自 code_context。

3. 不允许编造不存在的文件、函数、组件、
   API、状态、调用关系或行为。

4. 不允许因为某个文件 Retrieval Rank 高，
   就直接认为它应该修改。

5. evidence 必须说明实际观察到的代码事实。

好的 Evidence 示例：

- users store 定义 pageSize，默认值为 10
- setPageSize 同时把 page 重置为 1
- UsersView 调用 users store 获取分页状态
- listUsers 接收 UserQuery 并直接作为请求参数
- PaginationBar 当前只处理 page 和 totalPages

不好的 Evidence：

- 这个文件非常相关
- 它排名很高
- 应该修改这个文件
- 看起来可能有关

6. observed_role 定义：

candidate_target

    当前代码事实表明：
    这个文件很可能是后续实现需求时
    需要修改的位置。

    注意：
    这仍然只是候选，
    不是最终修改决定。

dependency

    后续实现需要理解这个文件，
    但当前 Evidence 更支持：
    它属于依赖关系或现有基础能力，
    原则上不应该主动修改。

reference

    当前文件提供类似实现、
    邻接实现、UI 模式或工程模式，
    主要用于参考。

test

    当前文件是与需求相关的测试文件。

unknown

    当前上下文不足以确定文件作用。

7. 如果 Requirement / TechnicalSpec
   明确声明某个模块不应修改，
   但当前代码仍需要理解该模块，
   应优先考虑 dependency，
   而不是 candidate_target。

8. confidence 范围为 0~1。

它表达 Evidence 强弱，
不是严格数学概率。

建议：

0.90 ~ 1.00
    非常明确的代码事实

0.70 ~ 0.89
    Evidence 较强

0.40 ~ 0.69
    有一定 Evidence，但仍存在不确定性

0.00 ~ 0.39
    Evidence 较弱

9. findings 只能记录已经从代码确认的事实。

10. unresolved_questions
    用来记录当前代码上下文仍无法确认的问题。

11. 如果某个问题需要读取未提供的文件才能回答，
    不要猜测，
    放入 unresolved_questions。

12. 不得声称已经修改代码。

13. 不得声称已经运行测试。

14. 不得读取或使用 Benchmark Oracle、
    expected_files 或任何标准答案。

15. 只返回 JSON。

16. 不返回 Markdown。

17. 返回结果必须严格满足以下 JSON Schema：

{json.dumps(
    schema,
    ensure_ascii=False,
    indent=2,
)}
""".strip()

    @staticmethod
    def _build_payload(
        *,
        bundle: TaskContextBundle,
    ) -> dict:

        return {
            "context_type": ("repository_investigation"),
            "task": (bundle.task.model_dump(mode="json")),
            "requirement": {
                "title": (bundle.requirement.title),
                "summary": (bundle.requirement.summary),
                "goals": list(bundle.requirement.goals),
                "non_goals": list(bundle.requirement.non_goals),
                "acceptance_criteria": (list(bundle.requirement.acceptance_criteria)),
                "constraints": list(bundle.requirement.constraints),
            },
            "technical_spec": {
                "summary": (bundle.technical_spec.summary),
                "solution_approach": (list(bundle.technical_spec.solution_approach)),
                "affected_modules": (list(bundle.technical_spec.affected_modules)),
                "reuse_plan": list(bundle.technical_spec.reuse_plan),
                "data_flow": list(bundle.technical_spec.data_flow),
                "technical_decisions": [
                    decision.model_dump(mode="json")
                    for decision in (bundle.technical_spec.technical_decisions)
                ],
                "open_questions": (list(bundle.technical_spec.open_questions)),
            },
            "repository": (bundle.repository.model_dump(mode="json")),
            "engineering_rules": list(bundle.engineering_rules),
            "code_context": [
                {
                    "relative_path": (item.relative_path),
                    "category": (item.category),
                    "language": (item.language),
                    "content_mode": (item.content_mode),
                    "start_line": (item.start_line),
                    "end_line": (item.end_line),
                    "relevance_reasons": (list(item.relevance_reasons)),
                    "content": (item.content),
                }
                for item in bundle.code_context
            ],
            "context_budget": (bundle.budget.model_dump(mode="json")),
        }

    @staticmethod
    def _normalize_path(
        path: str,
    ) -> str:

        normalized = path.replace(
            "\\",
            "/",
        ).strip()

        while normalized.startswith("./"):
            normalized = normalized[2:]

        return normalized

    def _canonicalize_paths(
        self,
        *,
        artifact: InvestigationArtifact,
        bundle: TaskContextBundle,
    ) -> InvestigationArtifact:
        """
        LLM 有时会：

            src\\stores\\users.ts

        而 Repository Context 使用：

            src/stores/users.ts

        这里只处理路径格式差异。

        不允许把未知路径“猜成”
        某个已知路径。
        """

        canonical_paths = {
            self._normalize_path(item.relative_path).casefold(): (item.relative_path)
            for item in bundle.code_context
        }

        normalized_files: list[ObservedRepositoryFile] = []

        for observed in artifact.observed_files:

            normalized = self._normalize_path(observed.relative_path)

            canonical = canonical_paths.get(normalized.casefold())

            if canonical is None:
                normalized_files.append(
                    observed.model_copy(
                        deep=True,
                        update={
                            "relative_path": (normalized),
                        },
                    )
                )

                continue

            normalized_files.append(
                observed.model_copy(
                    deep=True,
                    update={
                        "relative_path": (canonical),
                    },
                )
            )

        return artifact.model_copy(
            deep=True,
            update={
                "observed_files": (normalized_files),
            },
        )

    def _validate_output(
        self,
        *,
        artifact: InvestigationArtifact,
        bundle: TaskContextBundle,
    ) -> None:

        task = bundle.task

        if artifact.task_id != task.task_id:
            raise (
                InvestigationArtifactBuildError(
                    "InvestigationArtifact "
                    "task_id 与当前 Task 不一致："
                    f"expected={task.task_id}, "
                    f"actual={artifact.task_id}"
                )
            )

        allowed_paths = {
            self._normalize_path(item.relative_path).casefold()
            for item in bundle.code_context
        }

        invalid_paths: list[str] = []

        for observed in artifact.observed_files:
            normalized = self._normalize_path(observed.relative_path).casefold()

            if normalized not in allowed_paths:
                invalid_paths.append(observed.relative_path)

        if invalid_paths:
            raise (
                InvestigationArtifactBuildError(
                    "InvestigationArtifact "
                    "引用了未提供给模型的文件：" + ", ".join(invalid_paths)
                )
            )
