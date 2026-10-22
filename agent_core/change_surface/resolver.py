import json
from collections import defaultdict

from pydantic import ValidationError

from agent_core.change_surface.errors import ChangeSurfaceResolutionError
from agent_core.change_surface.model import (
    InvestigationArtifact,
    ResolvedChangeSurface,
)
from agent_core.codebase.model import HybridRelevantFileSelection
from agent_core.context.model import TaskContextBundle
from agent_core.llm.base import LLM
from agent_core.llm.model import ChatRequest, Message
from agent_core.artifacts.execution_model import (
    TaskExecutionArtifact,
)


class ChangeSurfaceResolver:
    """
    把 Planner 假设 + Retrieval + Investigation Facts
    解析成真实执行范围。

    关键原则：
    - Retrieval 相关性不等于修改权限；
    - code_change 的生产修改目标必须有上游 Investigation candidate_target 证据；
    - test Task 只能修改测试文件，生产代码只能作为 dependency/reference；
    - code_change 可以被 Repository Evidence 判定为 no-op；
    - Runtime 会对模型角色结果做确定性归一化与校验。
    """

    def __init__(
        self,
        *,
        llm: LLM,
        max_validation_retries: int = 1,
        min_candidate_confidence: float = 0.70,
    ) -> None:
        if max_validation_retries < 0:
            raise ValueError("max_validation_retries 不能小于 0")
        if not 0.0 <= min_candidate_confidence <= 1.0:
            raise ValueError("min_candidate_confidence 必须位于 0~1")

        self.llm = llm
        self.max_validation_retries = max_validation_retries
        self.min_candidate_confidence = min_candidate_confidence

    async def resolve(
        self,
        *,
        bundle: TaskContextBundle,
        selection: HybridRelevantFileSelection,
        upstream_investigations: list[InvestigationArtifact],
        upstream_executions: list[TaskExecutionArtifact] | None = None,
    ) -> ResolvedChangeSurface:
        execution_artifacts = list(upstream_executions or [])
        if bundle.task.task_type not in {"code_change", "test"}:
            raise ChangeSurfaceResolutionError(
                "ChangeSurfaceResolver 只处理 code_change / test Task"
            )

        schema = ResolvedChangeSurface.model_json_schema()
        available_paths = self._available_paths(
            bundle=bundle,
            selection=selection,
            investigations=upstream_investigations,
        )
        category_by_path = self._category_by_path(
            bundle=bundle,
            selection=selection,
        )
        investigation_roles = self._investigation_roles(upstream_investigations)
        candidate_targets = self._candidate_target_paths(upstream_investigations)

        payload = {
            "task": bundle.task.model_dump(mode="json"),
            "requirement": bundle.requirement.model_dump(mode="json"),
            "technical_spec": bundle.technical_spec.model_dump(mode="json"),
            "retrieved_files": [
                {
                    "relative_path": item.relative_path,
                    "category": item.category,
                    "context_role": item.context_role,
                    "matched_intents": list(item.matched_intents),
                    "target_match_type": item.target_match_type,
                    "target_match_value": item.target_match_value,
                    "rerank_rank": item.rerank_rank,
                    "rerank_score": item.rerank_score,
                    "reasons": list(item.reasons),
                }
                for item in selection.selected_files
            ],
            "upstream_investigations": [
                item.model_dump(mode="json") for item in upstream_investigations
            ],
            "upstream_executions": [
                item.model_dump(mode="json") for item in execution_artifacts
            ],
            "available_paths": sorted(available_paths),
            "runtime_candidate_targets": sorted(candidate_targets),
        }

        request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content=self._build_system_prompt(schema=schema),
                ),
                Message(
                    role="user",
                    content=json.dumps(payload, ensure_ascii=False, indent=2),
                ),
            ]
        )

        response = await self.llm.chat(request)
        raw_output = response.content
        last_error: Exception | None = None

        for attempt in range(self.max_validation_retries + 1):
            try:
                surface = self._parse_surface(raw_output)
                surface = self._normalize_task_roles(
                    surface=surface,
                    task_type=bundle.task.task_type,
                    category_by_path=category_by_path,
                )
                surface = self._enforce_investigation_boundary(
                    surface=surface,
                    task_type=bundle.task.task_type,
                    candidate_targets=candidate_targets,
                    investigation_roles=investigation_roles,
                )
                surface = self._derive_execution_disposition(
                    surface=surface,
                    task_type=bundle.task.task_type,
                )

                self._validate_surface(
                    surface=surface,
                    expected_task_id=bundle.task.task_id,
                    available_paths=available_paths,
                    task_type=bundle.task.task_type,
                    candidate_targets=candidate_targets,
                    category_by_path=category_by_path,
                )

                return surface

            except (
                ValidationError,
                json.JSONDecodeError,
                ValueError,
                ChangeSurfaceResolutionError,
            ) as exc:
                last_error = exc

                if attempt >= self.max_validation_retries:
                    break

                raw_output = await self._repair_output(
                    raw_output=raw_output,
                    error=exc,
                    schema=schema,
                    expected_task_id=bundle.task.task_id,
                    task_type=bundle.task.task_type,
                    available_paths=available_paths,
                    candidate_targets=candidate_targets,
                )

        error_detail = self._format_error(last_error)
        output_preview = self._preview(raw_output)

        raise ChangeSurfaceResolutionError(
            "LLM 返回的 ResolvedChangeSurface 在自动修复后仍不满足契约。\n"
            f"validation_error: {error_detail}\n"
            f"raw_output_preview: {output_preview}"
        ) from last_error

    @staticmethod
    def _build_system_prompt(
        *,
        schema: dict,
    ) -> str:
        return f"""
你是 Change Surface Resolver。
你的工作是根据当前 Task、Repository Retrieval Evidence 和上游 InvestigationArtifact，
区分：真正要修改的生产文件、只读依赖、参考文件、需要修改的测试文件，以及证据不足的问题。

核心语义：
- implementation_targets = 生产实现代码的修改目标。
- tests = 测试代码的修改目标。
- dependencies = 必须理解但默认只读。
- references = 类似/邻接实现，仅作参考。

硬性规则：
1. 所有角色中的文件都只能来自 available_paths。
2. 同一个 relative_path 只能出现在一个角色列表中。
3. 不要把“相关”“可能需要”“最可能”“如果需要”直接等价成“必须修改”。
4. code_change Task 的 implementation_targets 必须有上游 InvestigationArtifact 的 candidate_target 事实支持。
5. candidate_target 只是修改资格（eligibility），不是修改义务（obligation）。
   即使文件存在 candidate_target 证据，
   仍然必须判断当前 Task 是否真的需要修改它。
   Minimal Change 不等于最少文件数量。

    确定 implementation_targets 时必须维持现有 Responsibility Boundary：
    
    1. 如果 candidate_target 是 Requirement 所改变状态、
       行为或不变量的 owner，
       不要仅因为调用方可以绕过 owner 直接完成行为，
       就将 owner 从 implementation_targets 移除。
    
    2. 不得为了减少修改文件数量，
       将状态转换、不变量维护或 domain behavior
       泄漏到原本只负责 orchestration / presentation 的调用方。
    
    3. 最小必要修改集合应同时满足：
       - Requirement 行为完整；
       - 现有模块职责保持一致；
       - 状态不变量由其 owner 维护；
       - 不引入重复或平行实现。
6. implementation_targets 应是满足当前 Task 的最小必要修改集合。
   如果修改另一个已确认 target 即可完成当前 Task，
   不要因为某文件“也可以修改”就把它加入 implementation_targets。
7. 对 UI 组合关系：
   如果父级页面能够通过组合现有子组件并新增同级 UI 完成需求，
   不应仅因子组件位于相关区域就修改子组件。
8. 如果某个文件没有 candidate_target 证据，只能放 dependencies/references，或写入 unresolved；不要猜成修改目标。
9. 如果 Repository Evidence 已证明当前 Task 目标由现有实现满足，允许 implementation_targets 为空。不要为了 task_type=code_change 强行找文件修改。
10. 如果是否需要修改仍缺少关键代码证据，implementation_targets 可以为空，并把缺失事实放 unresolved。
11. test Task：测试文件只能放 tests；生产代码只能作为 dependencies/references，不能放 implementation_targets。
12. code_change Task：测试文件不能放 implementation_targets；如确实需要同步修改测试，放 tests。
13. Requirement non_goals / constraints 明确禁止修改的区域，不得进入 implementation_targets 或 tests。
14. API/Mock/Router 等若需求明确“不修改”，但理解实现需要它们，放 dependencies。
15. 优先使用 InvestigationArtifact 中的实际代码事实，而不是 rerank rank。
16. unresolved 只记录真正影响执行决策的未知事实，不重复改写同一个问题。
17. rationale 要说明为什么一个文件是 target/dependency/reference/test，禁止使用“排名高所以修改”。
18. execution_disposition 可以留 null；Runtime 会根据最终角色和 unresolved 重新计算。
19. 不得读取 Benchmark Oracle。
20. 只能输出 JSON 对象。不要 Markdown、不要 ```json code fence、不要解释文字。
21. 只能使用 Schema 中存在的字段；不要添加额外字段。
22. upstream_executions 是已经由 Runtime 确认 completed、
    applied 且 verification passed 的事实，
    它比 Planner 对“上游将会做什么”的假设更可靠。

23. upstream_executions.changed_files 只能证明：
    “某个上游 Task 已经真实修改并验证了这个文件”；
    不代表当前 Task 也必须再次修改该文件。

24. 如果 upstream_executions 已证明某个依赖 Task 已执行，
    不要再把“该 Task 是否已经执行”写入 unresolved。

25. Execution Artifact 只能确认修改事实和验证状态。
    对修改后的具体函数名、DOM、接口或行为，
    仍应优先依据当前 code_context 中的真实代码，
    不得凭 changed_files 猜测实现细节。

26. 如果 InvestigationArtifact 与 Execution Artifact
    对上游状态存在时间上的冲突，
    Execution Artifact 表示更晚的已完成状态，
    应优先用于判断上游 Task 是否已经执行。
JSON Schema：
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()

    @staticmethod
    def _extract_json_text(
        raw_output: str,
    ) -> str:
        text = (raw_output or "").strip()

        if not text:
            raise ValueError("LLM 返回空内容")

        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end < start:
            raise ValueError("LLM 输出中没有完整 JSON 对象")

        return text[start : end + 1]

    @classmethod
    def _parse_surface(
        cls,
        raw_output: str,
    ) -> ResolvedChangeSurface:
        json_text = cls._extract_json_text(raw_output)
        data = json.loads(json_text)

        if not isinstance(data, dict):
            raise ValueError("ResolvedChangeSurface 顶层必须是 JSON object")

        return ResolvedChangeSurface.model_validate(data)

    async def _repair_output(
        self,
        *,
        raw_output: str,
        error: Exception,
        schema: dict,
        expected_task_id: str,
        task_type: str,
        available_paths: set[str],
        candidate_targets: set[str],
    ) -> str:
        repair_request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content=f"""
你正在修复一个 ResolvedChangeSurface JSON。

上一次输出没有通过 Runtime Validation。
你只能修复 JSON 契约和文件角色，不能扩大任务范围。

硬性规则：
1. task_id 必须严格等于 {expected_task_id!r}。
2. 当前 task_type={task_type!r}。
3. 所有文件必须来自 available_paths。
4. code_change 的 implementation_targets 只能来自 runtime_candidate_targets。
5. test Task 的测试文件放 tests，生产代码不得放 implementation_targets。
6. 同一个文件只能属于 implementation_targets/dependencies/references/tests 中一个角色。
7. 没有足够证据时，从修改目标移除，并放 dependencies/references；确实阻塞执行时再写 unresolved。
8. 现有实现已满足 Task 时，允许没有任何 implementation target。
9. 不要添加 Schema 外字段。
10. 只返回 JSON，不要 Markdown，不要解释。

JSON Schema：
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip(),
                ),
                Message(
                    role="user",
                    content=json.dumps(
                        {
                            "validation_error": self._format_error(error),
                            "available_paths": sorted(available_paths),
                            "runtime_candidate_targets": sorted(candidate_targets),
                            "previous_output": raw_output,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
            ]
        )

        response = await self.llm.chat(repair_request)
        return response.content

    @staticmethod
    def _available_paths(
        *,
        bundle: TaskContextBundle,
        selection: HybridRelevantFileSelection,
        investigations: list[InvestigationArtifact],
    ) -> set[str]:
        paths = {item.relative_path for item in bundle.code_context}
        paths.update(item.relative_path for item in selection.selected_files)

        for artifact in investigations:
            paths.update(item.relative_path for item in artifact.observed_files)

        return paths

    @staticmethod
    def _category_by_path(
        *,
        bundle: TaskContextBundle,
        selection: HybridRelevantFileSelection,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for item in selection.selected_files:
            result[item.relative_path] = item.category
        for item in bundle.code_context:
            result[item.relative_path] = item.category
        return result

    @staticmethod
    def _investigation_roles(
        investigations: list[InvestigationArtifact],
    ) -> dict[str, set[str]]:
        roles: dict[str, set[str]] = defaultdict(set)
        for artifact in investigations:
            for item in artifact.observed_files:
                roles[item.relative_path].add(item.observed_role)
        return roles

    def _candidate_target_paths(
        self,
        investigations: list[InvestigationArtifact],
    ) -> set[str]:
        return {
            item.relative_path
            for artifact in investigations
            for item in artifact.observed_files
            if item.observed_role == "candidate_target"
            and item.confidence >= self.min_candidate_confidence
        }

    @classmethod
    def _normalize_task_roles(
        cls,
        *,
        surface: ResolvedChangeSurface,
        task_type: str,
        category_by_path: dict[str, str],
    ) -> ResolvedChangeSurface:
        implementation_targets: list[str] = []
        dependencies = list(surface.dependencies)
        references = list(surface.references)
        tests = list(surface.tests)
        rationale = list(surface.rationale)

        for path in surface.implementation_targets:
            if cls._is_test_file(path, category_by_path):
                tests.append(path)
                rationale.append(
                    f"runtime-role-normalization: {path} 是测试文件，"
                    "从 implementation_targets 移入 tests"
                )
                continue

            if task_type == "test":
                dependencies.append(path)
                rationale.append(
                    f"runtime-role-normalization: test Task 不修改生产文件 {path}，"
                    "降级为 dependency"
                )
                continue

            implementation_targets.append(path)

        return surface.model_copy(
            deep=True,
            update={
                "implementation_targets": cls._unique(implementation_targets),
                "dependencies": cls._unique(dependencies),
                "references": cls._unique(references),
                "tests": cls._unique(tests),
                "rationale": rationale,
            },
        )

    @classmethod
    def _enforce_investigation_boundary(
        cls,
        *,
        surface: ResolvedChangeSurface,
        task_type: str,
        candidate_targets: set[str],
        investigation_roles: dict[str, set[str]],
    ) -> ResolvedChangeSurface:
        if task_type != "code_change":
            return surface

        implementation_targets: list[str] = []
        dependencies = list(surface.dependencies)
        references = list(surface.references)
        tests = list(surface.tests)
        unresolved = list(surface.unresolved)
        rationale = list(surface.rationale)

        for path in surface.implementation_targets:
            if path in candidate_targets:
                implementation_targets.append(path)
                continue

            roles = investigation_roles.get(path, set())
            if "reference" in roles:
                references.append(path)
                downgraded_role = "reference"
            elif "test" in roles:
                tests.append(path)
                downgraded_role = "test"
            else:
                dependencies.append(path)
                downgraded_role = "dependency"

            rationale.append(
                "runtime-evidence-boundary: "
                f"{path} 缺少高置信度 Investigation candidate_target 证据，"
                f"从 implementation_target 降级为 {downgraded_role}"
            )
            unresolved.append(
                f"{path} 缺少足够 Repository Evidence 证明必须修改；"
                "如后续仍需要修改，应先补充 Investigation。"
            )

        return surface.model_copy(
            deep=True,
            update={
                "implementation_targets": cls._unique(implementation_targets),
                "dependencies": cls._unique(dependencies),
                "references": cls._unique(references),
                "tests": cls._unique(tests),
                "unresolved": cls._unique(unresolved),
                "rationale": rationale,
            },
        )

    @staticmethod
    def _derive_execution_disposition(
        *,
        surface: ResolvedChangeSurface,
        task_type: str,
    ) -> ResolvedChangeSurface:
        has_change = bool(
            surface.tests if task_type == "test" else surface.implementation_targets
        )

        if has_change:
            disposition = "change_required"
        elif surface.unresolved:
            disposition = "insufficient_evidence"
        else:
            disposition = "no_change_required"

        return surface.model_copy(
            deep=True,
            update={"execution_disposition": disposition},
        )

    @classmethod
    def _validate_surface(
        cls,
        *,
        surface: ResolvedChangeSurface,
        expected_task_id: str,
        available_paths: set[str],
        task_type: str,
        candidate_targets: set[str],
        category_by_path: dict[str, str],
    ) -> None:
        if surface.task_id != expected_task_id:
            raise ChangeSurfaceResolutionError(
                f"Surface task_id 不一致: expected={expected_task_id}, "
                f"actual={surface.task_id}"
            )

        used = {
            *surface.implementation_targets,
            *surface.dependencies,
            *surface.references,
            *surface.tests,
        }
        invalid = sorted(used - available_paths)
        if invalid:
            raise ChangeSurfaceResolutionError(
                "ResolvedChangeSurface 引用了未知文件: " + ", ".join(invalid)
            )

        if task_type == "test" and surface.implementation_targets:
            raise ChangeSurfaceResolutionError(
                "test Task 不允许包含生产 implementation_targets"
            )

        test_targets = [
            path
            for path in surface.implementation_targets
            if cls._is_test_file(path, category_by_path)
        ]
        if test_targets:
            raise ChangeSurfaceResolutionError(
                "测试文件不能出现在 implementation_targets: "
                + ", ".join(sorted(test_targets))
            )

        if task_type == "code_change":
            unsupported = sorted(
                set(surface.implementation_targets) - candidate_targets
            )
            if unsupported:
                raise ChangeSurfaceResolutionError(
                    "implementation_targets 缺少 Investigation candidate_target 证据: "
                    + ", ".join(unsupported)
                )

    @staticmethod
    def _is_test_file(
        path: str,
        category_by_path: dict[str, str],
    ) -> bool:
        normalized = path.replace("\\", "/").casefold()
        return (
            category_by_path.get(path) == "test"
            or normalized.startswith("tests/")
            or "/tests/" in normalized
            or normalized.endswith(".spec.ts")
            or normalized.endswith(".test.ts")
            or normalized.endswith(".spec.tsx")
            or normalized.endswith(".test.tsx")
            or normalized.endswith(".spec.js")
            or normalized.endswith(".test.js")
        )

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        return list(dict.fromkeys(item for item in items if item))

    @staticmethod
    def _format_error(
        error: Exception | None,
    ) -> str:
        if error is None:
            return "unknown validation error"

        if isinstance(error, ValidationError):
            details = []
            for item in error.errors(include_url=False):
                location = ".".join(str(part) for part in item.get("loc", ()))
                message = item.get("msg", "validation error")
                error_type = item.get("type", "unknown")
                details.append(f"{location or '<root>'}: {message} [{error_type}]")
            return "; ".join(details)

        return str(error)

    @staticmethod
    def _preview(
        text: str,
        *,
        limit: int = 4000,
    ) -> str:
        normalized = (text or "").strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit] + "...<truncated>"
