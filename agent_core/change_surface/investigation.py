import json

from pydantic import ValidationError

from agent_core.change_surface.errors import InvestigationArtifactBuildError
from agent_core.change_surface.model import (
    InvestigationArtifact,
    ObservedRepositoryFile,
)
from agent_core.context.model import TaskContextBundle
from agent_core.llm.base import LLM
from agent_core.llm.model import ChatRequest, Message


class InvestigationArtifactBuilder:
    """把 Investigation 的真实代码上下文转成结构化 Repository Facts。"""

    def __init__(self, *, llm: LLM) -> None:
        self.llm = llm

    async def build(self, *, bundle: TaskContextBundle) -> InvestigationArtifact:
        self._validate_input(bundle=bundle)
        schema = InvestigationArtifact.model_json_schema()
        request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content=self._build_system_prompt(schema=schema),
                ),
                Message(
                    role="user",
                    content=json.dumps(
                        self._build_payload(bundle=bundle),
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
            ]
        )
        response = await self.llm.chat(request)
        content = response.content.strip()
        if not content:
            raise InvestigationArtifactBuildError("LLM 返回空 InvestigationArtifact")

        try:
            artifact = InvestigationArtifact.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            raise InvestigationArtifactBuildError(
                "LLM 返回的 InvestigationArtifact 不满足 Schema"
            ) from exc

        artifact = self._canonicalize_paths(artifact=artifact, bundle=bundle)
        self._validate_output(artifact=artifact, bundle=bundle)
        return artifact

    @staticmethod
    def _validate_input(*, bundle: TaskContextBundle) -> None:
        if bundle.task.task_type != "investigation":
            raise InvestigationArtifactBuildError(
                "InvestigationArtifactBuilder 只能处理 investigation Task"
            )
        if not bundle.code_context:
            raise InvestigationArtifactBuildError(
                "Investigation Task 没有 Repository Code Context"
            )

    @staticmethod
    def _build_system_prompt(*, schema: dict) -> str:
        return f"""
你是一名资深软件仓库分析工程师。
你的任务不是修改代码，而是基于提供的 Repository Code Context 生成 InvestigationArtifact。

规则：
1. observed_files.relative_path 只能来自输入 code_context。
2. evidence 必须是代码事实，不能写“排名高”“很相关”等检索结论。
3. candidate_target 表示：
   基于当前提供的代码事实，为满足 Requirement，
   该文件中的现有职责/状态/行为本身存在必须修改的证据。
4. “可以在这里实现”“与功能相邻”“修改这里也能完成需求”
   不足以标记 candidate_target。
5. 如果 Requirement 可以通过修改父级组合层完成，
   而无需改变子组件自身契约，
   子组件应标记为 dependency，而不是 candidate_target。
6. 如果存在多个可行实现位置，但代码事实不足以证明某一个必须修改，
   不要把所有位置都标记为 candidate_target。
   保守标记 dependency，并把设计选择写入 unresolved_questions。
7. candidate_target 应遵循 Minimal Change Principle：
   在满足需求的前提下，优先识别最小必要生产修改面。
   candidate_target 不仅包括“代码无法完成需求而必须直接修改”的文件，
    也包括其现有职责所拥有的状态、行为或不变量因 Requirement
    发生变化而需要扩展的文件。
    
    判断 candidate_target 时必须考虑 Responsibility Ownership：
    
    1. 如果一个文件拥有 Requirement 正在改变的核心状态或业务不变量，
       即使调用方可以通过直接操作内部/公开状态绕过修改该文件，
       该 owner 仍应优先视为 candidate_target。
    
    2. 不要为了减少修改文件数量，
       把本应由状态 owner / domain owner 维护的行为泄漏到调用方。
    
    3. 如果已有模块已经通过 action / method 封装同类状态转换，
       新增同类状态转换应优先延续这一职责边界。
    
    4. “父组件可以直接修改子模块状态”不能单独作为
       排除该状态 owner candidate_target 的理由。
    
    5. Minimal Change Principle 的含义是：
       在维持现有架构职责、封装边界和状态不变量的前提下，
       选择最小必要修改集合；
       不是单纯追求修改文件数量最少。
    
    6. 对相邻 UI 组件仍保持保守：
       只有其自身 contract / behavior 必须变化时才是 candidate_target。
       仅仅因为 UI 位于该组件附近，不足以成为 candidate_target。
    7.1 必须区分“Domain Contract Change”
    和“Existing Contract Consumption”。

    Domain Contract Change：
    Requirement 改变了某个领域模型本身的
    字段、允许值、语义、不变量或状态转换规则。

    例如：
        原状态 union 不包含 refunded，
        Requirement 要新增 refunded。

    这种情况下，
    对应 domain/type owner 可以是 candidate_target。

    7.2 Existing Contract Consumption：
        Requirement 只是让新的 UI、查询、筛选、
        API 或业务流程使用一个已经存在且足够表达需求的
        字段或类型。
    
        如果当前 domain/type 已经完整覆盖
        Requirement 所需的所有值和语义，
        不要仅因为新功能会引用该类型，
        就把它标记为 candidate_target。
    
        这种文件应优先标记为 dependency。
    
    7.3 不得仅为了以下原因把现有 domain/type 文件
        标成 candidate_target：
    
        - 提取一个新的 type alias；
        - 为减少重复字符串创建类型；
        - 为代码更“整洁”增加 Query interface；
        - 为未来扩展提前抽象；
        - 当前实现“不够优雅”；
        - 多个文件都可能引用相同 literal。
    
        除非 Requirement、Engineering Rules、
        现有架构约定或当前代码契约明确证明
        该抽象是完成需求所必需的。
    
    7.4 candidate_target 必须回答：
    
        “如果这个文件保持当前内容完全不变，
         Requirement 是否仍然可以在维持现有职责边界的前提下
         正确完成？”
    
        如果答案是“可以”，
        则该文件原则上不应成为 candidate_target。
    
    7.5 Responsibility Ownership 规则不能被扩大解释为：
    
        “只要某个文件定义了需求中出现的数据类型，
         它就是修改目标”。
    
        Ownership 只有在该 owner 自身维护的
        contract / state / invariant / behavior
        因 Requirement 发生变化时，
        才形成修改义务。
8. target_match_type、relevance_reasons、rerank rank、
   directory match、Graph relation 都只能帮助定位代码，
   不能作为 candidate_target 的直接证据。
9. dependency 表示需要理解但原则上不主动修改。
10. reference 表示类似实现或邻接实现。
11. test 表示测试文件。
12. unknown 表示证据不足。
13. Requirement/TechnicalSpec 明确声明“不修改”的模块，如仍需理解，应优先标 dependency。
14. findings 只能记录已从代码确认的事实。
15. 未提供文件才能回答的问题进入 unresolved_questions，不要猜。
16. 不得声称已修改代码或已运行测试。
17. 不得读取或使用 Benchmark Oracle / expected_files。
18. 只返回 JSON，不返回 Markdown。
19. upstream_artifacts 中 artifact_type=execution 的 Artifact，
    表示对应上游 Task 已经实际 Apply 且 Verification Passed，
    可以用于确认上游任务是否已经真实完成。

20. Execution Artifact 中的 changed_files
    只证明文件在上游 Task 中已经被修改并验证，
    不代表当前 Investigation 应将该文件标记为 candidate_target。

21. 对上游修改后的具体代码事实，
    仍以当前提供的 code_context 为准；
    如果代码内容没有提供，不要仅根据文件名猜测具体实现。
    
JSON Schema：
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()

    @staticmethod
    def _build_payload(*, bundle: TaskContextBundle) -> dict:
        return {
            "context_type": "repository_investigation",
            "task": bundle.task.model_dump(mode="json"),
            "requirement": {
                "title": bundle.requirement.title,
                "summary": bundle.requirement.summary,
                "goals": list(bundle.requirement.goals),
                "non_goals": list(bundle.requirement.non_goals),
                "acceptance_criteria": list(bundle.requirement.acceptance_criteria),
                "constraints": list(bundle.requirement.constraints),
            },
            "technical_spec": {
                "summary": bundle.technical_spec.summary,
                "solution_approach": list(bundle.technical_spec.solution_approach),
                "affected_modules": list(bundle.technical_spec.affected_modules),
                "reuse_plan": list(bundle.technical_spec.reuse_plan),
                "data_flow": list(bundle.technical_spec.data_flow),
                "technical_decisions": [
                    item.model_dump(mode="json")
                    for item in bundle.technical_spec.technical_decisions
                ],
                "open_questions": list(bundle.technical_spec.open_questions),
            },
            "repository": bundle.repository.model_dump(mode="json"),
            "engineering_rules": list(bundle.engineering_rules),
            "upstream_artifacts": list(bundle.upstream_artifacts),
            "code_context": [
                {
                    "relative_path": item.relative_path,
                    "category": item.category,
                    "language": item.language,
                    "content_mode": item.content_mode,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "relevance_reasons": list(item.relevance_reasons),
                    "content": item.content,
                }
                for item in bundle.code_context
            ],
            "context_budget": bundle.budget.model_dump(mode="json"),
        }

    @staticmethod
    def _normalize_path(path: str) -> str:
        normalized = path.replace("\\", "/").strip()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    def _canonicalize_paths(
        self,
        *,
        artifact: InvestigationArtifact,
        bundle: TaskContextBundle,
    ) -> InvestigationArtifact:
        canonical_paths = {
            self._normalize_path(item.relative_path).casefold(): item.relative_path
            for item in bundle.code_context
        }
        normalized_files: list[ObservedRepositoryFile] = []
        for observed in artifact.observed_files:
            normalized = self._normalize_path(observed.relative_path)
            canonical = canonical_paths.get(normalized.casefold(), normalized)
            normalized_files.append(
                observed.model_copy(
                    deep=True,
                    update={"relative_path": canonical},
                )
            )
        return artifact.model_copy(
            deep=True,
            update={"observed_files": normalized_files},
        )

    def _validate_output(
        self,
        *,
        artifact: InvestigationArtifact,
        bundle: TaskContextBundle,
    ) -> None:
        if artifact.task_id != bundle.task.task_id:
            raise InvestigationArtifactBuildError(
                f"Artifact task_id 不一致: expected={bundle.task.task_id}, "
                f"actual={artifact.task_id}"
            )

        allowed_paths = {
            self._normalize_path(item.relative_path).casefold()
            for item in bundle.code_context
        }
        invalid = [
            item.relative_path
            for item in artifact.observed_files
            if self._normalize_path(item.relative_path).casefold() not in allowed_paths
        ]
        if invalid:
            raise InvestigationArtifactBuildError(
                "InvestigationArtifact 引用了未提供的文件: " + ", ".join(invalid)
            )
