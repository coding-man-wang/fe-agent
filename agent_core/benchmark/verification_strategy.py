from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.change_surface.model import (
    ResolvedChangeSurface,
)
from agent_core.codebase.model import (
    RepositoryProfile,
)
from agent_core.context.model import (
    CodeContextItem,
)
from agent_core.coding.handoff import (
    AnalysisArtifact,
)
from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
)
from agent_core.coding.patch_generator import (
    PatchGenerator,
)
from agent_core.coding.patch_model import (
    PatchPlan,
)
from agent_core.coding.preparation_model import (
    ExecutionPreparation,
)
from agent_core.policies.model import (
    EffectivePolicy,
    EngineeringRule,
)
from agent_core.sdd.model import (
    TaskDefinition,
)

TestLayer = Literal[
    "unit",
    "e2e",
    "mixed",
    "unknown",
]

RuntimePath = Literal[
    "local_mock",
    "remote_http",
    "mixed",
    "unknown",
]

CapabilityStatus = Literal[
    "available",
    "unavailable",
    "unproven",
]

ProofSource = Literal[
    "requirement",
    "technical_spec",
    "task_verification",
    "task_output",
]

ProofRoutingAction = Literal[
    "required_here",
    "allowed_here",
    "deferred",
]


# ============================================================
# Models
# ============================================================


class VerificationCapability(BaseModel):
    """
    当前 Test Layer 中一项真正可用的
    Verification Capability。

    Capability 不是 Requirement。

    Requirement:
        必须证明什么。

    Capability:
        当前环境能不能用这种方式证明。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    capability_id: str

    status: CapabilityStatus

    evidence: list[str] = Field(
        default_factory=list,
    )


class ProofRoute(BaseModel):
    """
    一个 Verification Obligation
    在当前 Test Layer 中应该如何处理。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    proof_id: str

    source: ProofSource

    description: str

    preferred_layer: TestLayer

    routing_action: ProofRoutingAction

    observable_in_current_layer: bool

    reason: str


class VerificationStrategyPlan(BaseModel):
    """
    Harness 计算出的当前 Test Task
    的 Verification Strategy。

    这是 Runtime Evidence，
    不是 LLM 猜测。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    task_id: str

    selected_layer: TestLayer

    runtime_path: RuntimePath

    writable_test_files: list[str] = Field(
        default_factory=list,
    )

    remote_endpoint_hints: list[str] = Field(
        default_factory=list,
    )

    capabilities: list[VerificationCapability] = Field(
        default_factory=list,
    )

    proof_routes: list[ProofRoute] = Field(
        default_factory=list,
    )

    allowed_strategies: list[str] = Field(
        default_factory=list,
    )

    forbidden_strategies: list[str] = Field(
        default_factory=list,
    )

    reasons: list[str] = Field(
        default_factory=list,
    )

    def required_proofs(
        self,
    ) -> list[ProofRoute]:
        return [
            item
            for item in self.proof_routes
            if (item.routing_action == "required_here")
        ]

    def deferred_proofs(
        self,
    ) -> list[ProofRoute]:
        return [
            item for item in self.proof_routes if (item.routing_action == "deferred")
        ]

    def capability(
        self,
        capability_id: str,
    ) -> VerificationCapability | None:

        for capability in self.capabilities:
            if capability.capability_id == capability_id:
                return capability

        return None

    def to_policy_rules(
        self,
    ) -> list[EngineeringRule]:
        """
        把 Runtime Strategy 转成
        PatchGenerator /
        VerificationRepair 都能消费的
        Dynamic Effective Policy。

        注意：
        Policy 只约束 HOW，
        不扩大 PatchPlan。
        """

        rules: list[EngineeringRule] = []

        required_lines = [
            (f"- [{item.proof_id}] " f"{item.description}")
            for item in self.required_proofs()
        ]

        deferred_lines = [
            (f"- [{item.proof_id}] " f"{item.description} " f"→ {item.preferred_layer}")
            for item in self.deferred_proofs()
        ]

        rules.append(
            EngineeringRule(
                rule_id=("TEST-STRATEGY-001"),
                category=("verification_strategy"),
                severity="must",
                source="runtime",
                description=(
                    "当前 Test Task 的 "
                    "Harness-selected layer 为 "
                    f"{self.selected_layer}；"
                    "测试实现必须服从当前 "
                    "verification_strategy，"
                    "不得为了方便自行切换到"
                    "当前环境不可观察的 Proof Strategy。"
                ),
            )
        )

        if required_lines:
            rules.append(
                EngineeringRule(
                    rule_id=("TEST-PROOF-REQUIRED-001"),
                    category=("verification_strategy"),
                    severity="must",
                    source="runtime",
                    description=(
                        "当前 Test Layer "
                        "必须优先证明以下 "
                        "Verification Obligations：\n" + "\n".join(required_lines)
                    ),
                )
            )

        if deferred_lines:
            rules.append(
                EngineeringRule(
                    rule_id=("TEST-PROOF-DEFERRED-001"),
                    category=("verification_strategy"),
                    severity="must",
                    source="runtime",
                    description=(
                        "以下 Verification Obligations "
                        "在当前 Test Layer "
                        "不可可靠观察，"
                        "必须保留其证明义务，"
                        "但不得在当前层伪造 Proof；"
                        "应该交给指定层验证：\n" + "\n".join(deferred_lines)
                    ),
                )
            )

        if "e2e_remote_network_interception" in self.forbidden_strategies:
            rules.append(
                EngineeringRule(
                    rule_id=("TEST-E2E-NETWORK-001"),
                    category=("verification_strategy"),
                    severity="must",
                    source="runtime",
                    description=(
                        "当前 Browser E2E "
                        "没有证据证明 Feature "
                        "实际走 Remote HTTP Runtime Path。"
                        "不得使用 page.route、"
                        "waitForRequest、waitForResponse、"
                        "request-count assertion "
                        "来证明该 Feature 的行为；"
                        "这些操作无法证明当前真实 Runtime Path。"
                    ),
                )
            )

        if self.runtime_path == "local_mock" and self.selected_layer == "e2e":
            rules.append(
                EngineeringRule(
                    rule_id=("TEST-E2E-LOCAL-MOCK-001"),
                    category=("verification_strategy"),
                    severity="must",
                    source="runtime",
                    description=(
                        "当前 E2E Runtime Path "
                        "由证据判断为 local_mock。"
                        "筛选切换、重新加载等行为"
                        "应通过用户操作后的可见结果变化、"
                        "loading/state transition "
                        "或最终页面状态证明；"
                        "不得把不存在的 HTTP request count "
                        "当成行为证明。"
                    ),
                )
            )

        if self.selected_layer == "unit":
            rules.append(
                EngineeringRule(
                    rule_id=("TEST-UNIT-CAPABILITY-001"),
                    category=("verification_strategy"),
                    severity="must",
                    source="runtime",
                    description=(
                        "当前测试层为 unit。"
                        "允许通过当前项目测试框架的 "
                        "module mock、dependency stub、"
                        "rejection injection "
                        "验证 API 参数映射、"
                        "dependency failure、error state "
                        "等不适合 Browser E2E "
                        "直接注入的 Contract。"
                    ),
                )
            )

        return rules


class VerificationStrategyViolation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    code: str

    relative_path: str

    message: str


class VerificationStrategyDecision(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    allowed: bool

    violations: list[VerificationStrategyViolation] = Field(
        default_factory=list,
    )

    def format_message(
        self,
    ) -> str:

        if not self.violations:
            return ""

        return " | ".join(
            (f"{item.code}: " f"{item.relative_path}: " f"{item.message}")
            for item in self.violations
        )


class StrategyAwarePatchGenerationResult(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    proposal: FilePatchProposal

    correction_count: int = Field(
        ge=0,
    )


class VerificationStrategyError(ValueError):
    pass


# ============================================================
# Resolver
# ============================================================


class VerificationStrategyResolver:
    """
    Deterministic Verification Strategy Resolver。

    不调用 LLM。

    输入：
        Current Repository Evidence
        + Test Writable Surface
        + Requirement / Technical / Task Proof Obligations

    输出：
        当前测试层真正具备哪些能力，
        什么应该在当前层证明，
        什么必须 deferred。
    """

    _REMOTE_CONTRACT_KEYWORDS = (
        "remote",
        "http",
        "query parameter",
        "request parameter",
        "api parameter",
        "api 参数",
        "接口参数",
        "请求参数",
        "远程",
        "参数映射",
        "get /",
        "post /",
        "put /",
        "delete /",
    )

    _FAILURE_KEYWORDS = (
        "error",
        "failure",
        "failed",
        "reject",
        "rejection",
        "exception",
        "错误",
        "失败",
        "异常",
        "报错",
    )

    _E2E_BEHAVIOR_KEYWORDS = (
        "loading",
        "router",
        "navigation",
        "navigate",
        "visible",
        "render",
        "filter",
        "switch",
        "reload",
        "refetch",
        "rapid",
        "page",
        "ui",
        "显示",
        "页面",
        "筛选",
        "切换",
        "重新加载",
        "重新请求",
        "快速",
        "路由",
        "加载",
    )

    _REMOTE_OVERRIDE_PATTERNS = (
        re.compile(
            r"VITE_[A-Z0-9_]*API_MODE" r"\s*=\s*[\"']?remote",
            re.IGNORECASE,
        ),
        re.compile(
            r"VITE_[A-Z0-9_]*USE_MOCK" r"\s*=\s*[\"']?false",
            re.IGNORECASE,
        ),
    )

    _LOCAL_DEFAULT_PATTERNS = (
        re.compile(
            r"VITE_[A-Z0-9_]*API_MODE" r"[^\n]{0,80}" r"!={1,2}\s*[\"']remote[\"']",
            re.IGNORECASE,
        ),
        re.compile(
            r"VITE_[A-Z0-9_]*USE_MOCK" r"[^\n]{0,80}" r"!={1,2}\s*[\"']false[\"']",
            re.IGNORECASE,
        ),
        re.compile(
            r"VITE_[A-Z0-9_]*API_MODE" r"[^\n]{0,80}" r"={2,3}\s*[\"']mock[\"']",
            re.IGNORECASE,
        ),
    )

    _REMOTE_ENDPOINT_PATTERN = re.compile(
        r"""
            (?:
                url\s*:\s*
                |
                fetch\s*\(\s*
                |
                request\s*\(\s*
            )
            ["']
            (?P<path>/[^"']+)
            ["']
            """,
        re.IGNORECASE | re.VERBOSE,
    )

    def resolve(
        self,
        *,
        repository_root: str | Path,
        task: TaskDefinition,
        repository: RepositoryProfile,
        code_context: Sequence[CodeContextItem],
        execution_surface: ResolvedChangeSurface,
        requirement_acceptance_criteria: Sequence[str] = (),
        technical_testing_strategy: Sequence[str] = (),
    ) -> VerificationStrategyPlan:

        root = Path(repository_root).resolve()

        selected_layer = self._select_layer(
            task=task,
            surface=(execution_surface),
        )

        code_text = "\n".join(item.content for item in code_context)

        runtime_config_text = self._runtime_config_text(
            root=root,
            repository=repository,
        )

        runtime_path = self._resolve_runtime_path(
            code_text=code_text,
            runtime_config_text=(runtime_config_text),
        )

        endpoints = self._extract_remote_endpoints(code_text)

        capabilities = self._build_capabilities(
            selected_layer=(selected_layer),
            runtime_path=(runtime_path),
            repository=repository,
        )

        proof_routes = self._build_proof_routes(
            task=task,
            selected_layer=(selected_layer),
            runtime_path=(runtime_path),
            requirement_acceptance_criteria=(requirement_acceptance_criteria),
            technical_testing_strategy=(technical_testing_strategy),
        )

        (
            allowed_strategies,
            forbidden_strategies,
        ) = self._strategy_sets(
            selected_layer=(selected_layer),
            runtime_path=(runtime_path),
            capabilities=capabilities,
        )

        writable_test_files = self._stable_unique(
            [
                *execution_surface.tests,
                *execution_surface.create_files,
            ]
        )

        reasons = [
            ("test layer 由当前 " "Execution Surface 的 writable " "test paths 决定"),
            (
                "runtime path 由 Current Revision "
                "Code Context + E2E runtime config "
                "确定性推导"
            ),
            (
                "不可观察的 Proof Obligation "
                "只允许路由到其他 Test Layer，"
                "不允许被删除或伪造"
            ),
        ]

        return VerificationStrategyPlan(
            task_id=task.task_id,
            selected_layer=selected_layer,
            runtime_path=runtime_path,
            writable_test_files=(writable_test_files),
            remote_endpoint_hints=endpoints,
            capabilities=capabilities,
            proof_routes=proof_routes,
            allowed_strategies=(allowed_strategies),
            forbidden_strategies=(forbidden_strategies),
            reasons=reasons,
        )

    # --------------------------------------------------------
    # Test Layer
    # --------------------------------------------------------

    def _select_layer(
        self,
        *,
        task: TaskDefinition,
        surface: ResolvedChangeSurface,
    ) -> TestLayer:

        paths = [
            *surface.tests,
            *surface.create_files,
        ]

        normalized = [
            path.replace(
                "\\",
                "/",
            ).lower()
            for path in paths
        ]

        has_unit = any("/unit/" in f"/{path}" for path in normalized)

        has_e2e = any("/e2e/" in f"/{path}" for path in normalized)

        if has_unit and has_e2e:
            return "mixed"

        if has_unit:
            return "unit"

        if has_e2e:
            return "e2e"

        fallback_text = (f"{task.title} " f"{task.objective}").lower()

        if "e2e" in fallback_text:
            return "e2e"

        if "unit" in fallback_text or "单元测试" in fallback_text:
            return "unit"

        return "unknown"

    # --------------------------------------------------------
    # Runtime Path
    # --------------------------------------------------------

    def _resolve_runtime_path(
        self,
        *,
        code_text: str,
        runtime_config_text: str,
    ) -> RuntimePath:

        lower = code_text.lower()

        has_local_mock = (
            ("usemock" in lower)
            or ("mocklist" in lower)
            or ("@/mocks/" in lower)
            or ("/mocks/" in lower)
        )

        has_remote_request = (
            "request<" in lower
            or "request(" in lower
            or "fetch(" in lower
            or "axios" in lower
        )

        explicit_remote = any(
            pattern.search(runtime_config_text) is not None
            for pattern in self._REMOTE_OVERRIDE_PATTERNS
        )

        local_default = any(
            pattern.search(code_text) is not None
            for pattern in self._LOCAL_DEFAULT_PATTERNS
        )

        if explicit_remote and has_remote_request:
            return "remote_http"

        if has_local_mock and local_default:
            return "local_mock"

        if has_local_mock and has_remote_request:
            return "mixed"

        if has_local_mock:
            return "local_mock"

        if has_remote_request:
            return "remote_http"

        return "unknown"

    def _runtime_config_text(
        self,
        *,
        root: Path,
        repository: RepositoryProfile,
    ) -> str:

        parts: list[str] = []

        for key, value in repository.scripts.items():
            parts.append(f"{key}={value}")

        candidates = [
            "package.json",
            "playwright.config.ts",
            "playwright.config.js",
            "playwright.config.mjs",
            "playwright.config.cjs",
            ".env",
            ".env.e2e",
            ".env.test",
        ]

        for name in candidates:

            path = root / name

            if not path.is_file():
                continue

            try:
                data = path.read_bytes()

            except OSError:
                continue

            if len(data) > 100_000:
                continue

            try:
                text = data.decode("utf-8")

            except UnicodeDecodeError:
                continue

            parts.append(text)

        return "\n".join(parts)

    def _extract_remote_endpoints(
        self,
        code_text: str,
    ) -> list[str]:

        result: list[str] = []

        for match in self._REMOTE_ENDPOINT_PATTERN.finditer(code_text):
            value = match.group("path").strip()

            if value and value not in result:
                result.append(value)

        return result

    # --------------------------------------------------------
    # Capabilities
    # --------------------------------------------------------

    def _build_capabilities(
        self,
        *,
        selected_layer: TestLayer,
        runtime_path: RuntimePath,
        repository: RepositoryProfile,
    ) -> list[VerificationCapability]:

        metadata = " ".join(
            [
                *repository.test_stack,
                *repository.dependencies,
                *repository.scripts.keys(),
                *repository.scripts.values(),
            ]
        ).lower()

        has_unit_framework = "vitest" in metadata or "jest" in metadata

        has_playwright = "playwright" in metadata

        result = [
            VerificationCapability(
                capability_id=("unit_module_mocking"),
                status=("available" if has_unit_framework else "unproven"),
                evidence=[
                    (
                        "unit test framework " "detected"
                        if has_unit_framework
                        else ("unit test framework " "not proven")
                    )
                ],
            ),
            VerificationCapability(
                capability_id=("dependency_failure_injection"),
                status=("available" if has_unit_framework else "unproven"),
                evidence=[
                    (
                        "dependency failure can " "be injected at unit layer"
                        if has_unit_framework
                        else ("dependency injection " "capability not proven")
                    )
                ],
            ),
            VerificationCapability(
                capability_id=("browser_ui_observation"),
                status=("available" if has_playwright else "unproven"),
                evidence=[
                    (
                        "Playwright detected"
                        if has_playwright
                        else ("browser E2E " "framework not proven")
                    )
                ],
            ),
        ]

        if selected_layer == "e2e" and runtime_path == "remote_http" and has_playwright:
            network_status: CapabilityStatus = "available"

            network_evidence = ["current E2E runtime " "is proven remote_http"]

        elif selected_layer == "e2e" and runtime_path == "local_mock":
            network_status = "unavailable"

            network_evidence = [
                (
                    "current feature path "
                    "is local_mock; "
                    "browser request interception "
                    "cannot prove this path"
                )
            ]

        else:
            network_status = "unproven"

            network_evidence = ["remote browser HTTP path " "is not proven"]

        result.append(
            VerificationCapability(
                capability_id=("browser_feature_" "network_interception"),
                status=network_status,
                evidence=network_evidence,
            )
        )

        result.append(
            VerificationCapability(
                capability_id=("local_mock_ui_observation"),
                status=(
                    "available"
                    if (
                        selected_layer == "e2e"
                        and runtime_path == "local_mock"
                        and has_playwright
                    )
                    else "unproven"
                ),
                evidence=[
                    (
                        "local mock behavior "
                        "can be observed through "
                        "rendered browser state"
                    )
                ],
            )
        )

        return result

    # --------------------------------------------------------
    # Verification Obligation Routing
    # --------------------------------------------------------

    def _build_proof_routes(
        self,
        *,
        task: TaskDefinition,
        selected_layer: TestLayer,
        runtime_path: RuntimePath,
        requirement_acceptance_criteria: Sequence[str],
        technical_testing_strategy: Sequence[str],
    ) -> list[ProofRoute]:

        candidates: list[tuple[ProofSource, str]] = []

        candidates.extend(
            (
                "requirement",
                value,
            )
            for value in (requirement_acceptance_criteria)
        )

        candidates.extend(
            (
                "technical_spec",
                value,
            )
            for value in (technical_testing_strategy)
        )

        candidates.extend(
            (
                "task_verification",
                value,
            )
            for value in task.verification
        )

        candidates.extend(
            (
                "task_output",
                value,
            )
            for value in task.expected_outputs
        )

        seen: set[str] = set()

        result: list[ProofRoute] = []

        for source, raw in candidates:

            description = raw.strip()

            if not description:
                continue

            normalized = description.lower()

            if normalized in seen:
                continue

            seen.add(normalized)

            preferred_layer = self._preferred_layer_for_proof(
                description=description,
                current_layer=(selected_layer),
                runtime_path=(runtime_path),
            )

            if preferred_layer == selected_layer:
                routing_action: ProofRoutingAction = "required_here"

                observable = True

                reason = "当前 Test Layer " "具备更合适的 " "Proof Capability"

            elif preferred_layer in {
                "unit",
                "e2e",
            } and selected_layer in {
                "unit",
                "e2e",
            }:
                routing_action = "deferred"

                observable = False

                reason = (
                    "该 Obligation "
                    f"更适合 {preferred_layer}；"
                    "当前层不得伪造 Proof"
                )

            else:
                routing_action = "allowed_here"

                observable = True

                reason = (
                    "当前规则无法进一步"
                    "细分 Test Layer，"
                    "允许当前层按已有"
                    "Repository Pattern 验证"
                )

            proof_id = self._proof_id(
                source=source,
                description=description,
            )

            result.append(
                ProofRoute(
                    proof_id=proof_id,
                    source=source,
                    description=(description),
                    preferred_layer=(preferred_layer),
                    routing_action=(routing_action),
                    observable_in_current_layer=(observable),
                    reason=reason,
                )
            )

        return result

    def _preferred_layer_for_proof(
        self,
        *,
        description: str,
        current_layer: TestLayer,
        runtime_path: RuntimePath,
    ) -> TestLayer:

        lower = description.lower()

        if self._contains_any(
            lower,
            self._REMOTE_CONTRACT_KEYWORDS,
        ):
            return "unit"

        if self._contains_any(
            lower,
            self._FAILURE_KEYWORDS,
        ):
            # v1.3D 保守策略：
            #
            # Failure Injection
            # 默认在 Unit 层证明。
            #
            # Browser E2E 只有明确的
            # Remote Runtime Capability
            # 才可能适合，但当前先不扩大。
            return "unit"

        if self._contains_any(
            lower,
            self._E2E_BEHAVIOR_KEYWORDS,
        ):
            return "e2e"

        if current_layer in {
            "unit",
            "e2e",
        }:
            return current_layer

        # unknown / mixed 时不强行猜。
        if runtime_path == "local_mock":
            return "unit"

        return "unknown"

    # --------------------------------------------------------
    # Strategy
    # --------------------------------------------------------

    def _strategy_sets(
        self,
        *,
        selected_layer: TestLayer,
        runtime_path: RuntimePath,
        capabilities: Sequence[VerificationCapability],
    ) -> tuple[
        list[str],
        list[str],
    ]:

        capability_map = {item.capability_id: item.status for item in capabilities}

        allowed: list[str] = []

        forbidden: list[str] = []

        if selected_layer == "unit":

            allowed.extend(
                [
                    "unit_module_mock",
                    ("unit_dependency_" "failure_injection"),
                    ("unit_api_contract_" "assertion"),
                    ("unit_component_" "behavior"),
                ]
            )

            forbidden.extend(
                [
                    "unit_browser_driver",
                ]
            )

        elif selected_layer == "e2e":

            allowed.extend(
                [
                    ("e2e_public_ui_" "observation"),
                    ("e2e_user_interaction"),
                    ("e2e_visible_state_" "transition"),
                    ("e2e_router_observation"),
                ]
            )

            if (
                capability_map.get(("browser_feature_" "network_interception"))
                == "available"
            ):
                allowed.append(("e2e_remote_network_" "interception"))

            else:
                forbidden.extend(
                    [
                        ("e2e_remote_network_" "interception"),
                        ("e2e_request_count_" "assertion"),
                        ("e2e_remote_failure_" "injection"),
                    ]
                )

            if runtime_path == "local_mock":
                allowed.append(("e2e_local_mock_" "visible_behavior"))

        return (
            self._stable_unique(allowed),
            self._stable_unique(forbidden),
        )

    # --------------------------------------------------------
    # Common
    # --------------------------------------------------------

    @staticmethod
    def _contains_any(
        text: str,
        values: Sequence[str],
    ) -> bool:

        return any(item in text for item in values)

    @staticmethod
    def _proof_id(
        *,
        source: str,
        description: str,
    ) -> str:

        digest = hashlib.sha256(
            (f"{source}:" f"{description}").encode("utf-8")
        ).hexdigest()[:10]

        return "PROOF-" + digest.upper()

    @staticmethod
    def _stable_unique(
        values: Sequence[str],
    ) -> list[str]:

        result: list[str] = []

        seen: set[str] = set()

        for value in values:

            if not value:
                continue

            if value in seen:
                continue

            seen.add(value)

            result.append(value)

        return result


# ============================================================
# Deterministic Proposal Guard
# ============================================================


class VerificationStrategyGuard:
    """
    Patch 生成之后、
    Side Effect 之前执行。

    Agent 可以选择具体测试代码，
    但不能使用 Harness 已证明
    当前 Runtime Path 不成立的
    Proof Strategy。
    """

    _NETWORK_PATTERNS = (
        (
            "E2E_PAGE_ROUTE",
            re.compile(
                r"\b(?:page|context)" r"\.route\s*\(",
            ),
        ),
        (
            "E2E_WAIT_FOR_REQUEST",
            re.compile(
                r"\bpage" r"\.waitForRequest\s*\(",
            ),
        ),
        (
            "E2E_WAIT_FOR_RESPONSE",
            re.compile(
                r"\bpage" r"\.waitForResponse\s*\(",
            ),
        ),
        (
            "E2E_REQUEST_LISTENER",
            re.compile(
                r"""
                \b(?:page|context)
                \.on\s*
                \(
                \s*["']request["']
                """,
                re.VERBOSE,
            ),
        ),
    )

    _UNIT_BROWSER_PATTERNS = (
        (
            "UNIT_PLAYWRIGHT_IMPORT",
            re.compile(r"@playwright/test"),
        ),
        (
            "UNIT_PAGE_GOTO",
            re.compile(r"\bpage\.goto\s*\("),
        ),
    )

    def evaluate(
        self,
        *,
        strategy: VerificationStrategyPlan | None,
        proposal: FilePatchProposal,
    ) -> VerificationStrategyDecision:

        if strategy is None:
            return VerificationStrategyDecision(
                allowed=True,
            )

        violations: list[VerificationStrategyViolation] = []

        for patch in proposal.patches:

            path = patch.relative_path.replace("\\", "/")

            content = patch.after_content

            lower_path = path.lower()

            # --------------------------------
            # Layer Boundary
            # --------------------------------

            if strategy.selected_layer == "unit" and "/e2e/" in f"/{lower_path}":
                violations.append(
                    VerificationStrategyViolation(
                        code=("TEST_LAYER_MISMATCH"),
                        relative_path=path,
                        message=("Unit Test Task " "不能生成 E2E test file"),
                    )
                )

            if strategy.selected_layer == "e2e" and "/unit/" in f"/{lower_path}":
                violations.append(
                    VerificationStrategyViolation(
                        code=("TEST_LAYER_MISMATCH"),
                        relative_path=path,
                        message=("E2E Test Task " "不能生成 Unit test file"),
                    )
                )

            # --------------------------------
            # E2E Network Capability
            # --------------------------------

            if strategy.selected_layer == "e2e" and (
                "e2e_remote_network_" "interception" in strategy.forbidden_strategies
            ):

                if not self._touches_feature_endpoint(
                    content=content,
                    endpoint_hints=(strategy.remote_endpoint_hints),
                ):
                    continue

                for (
                    code,
                    pattern,
                ) in self._NETWORK_PATTERNS:

                    if pattern.search(content) is None:
                        continue

                    violations.append(
                        VerificationStrategyViolation(
                            code=code,
                            relative_path=path,
                            message=(
                                "当前 E2E Runtime "
                                f"为 {strategy.runtime_path}，"
                                "没有证据证明 Feature "
                                "经过 Remote HTTP；"
                                "不能使用 browser network "
                                "interception/request count "
                                "作为当前行为证明"
                            ),
                        )
                    )

            # --------------------------------
            # Unit Layer Browser Driver
            # --------------------------------

            if strategy.selected_layer == "unit":
                for (
                    code,
                    pattern,
                ) in self._UNIT_BROWSER_PATTERNS:

                    if pattern.search(content) is None:
                        continue

                    violations.append(
                        VerificationStrategyViolation(
                            code=code,
                            relative_path=path,
                            message=(
                                "当前 Test Layer "
                                "为 unit，"
                                "不能使用 Browser "
                                "Driver 作为证明策略"
                            ),
                        )
                    )

        return VerificationStrategyDecision(
            allowed=(not violations),
            violations=violations,
        )

    @staticmethod
    def _touches_feature_endpoint(
        *,
        content: str,
        endpoint_hints: Sequence[str],
    ) -> bool:
        """
        避免把与当前 Feature 无关的
        page.route 一刀切禁止。

        只有当前测试源码引用了
        Current Repository 发现的
        Feature endpoint，
        才按 Feature Network Strategy
        做 deterministic block。
        """

        if not endpoint_hints:
            # 没有 endpoint Evidence 时，
            # Harness 不做过度推断；
            # 仍由 Dynamic Policy 约束模型。
            return False

        lower = content.lower()

        return any(endpoint.lower() in lower for endpoint in endpoint_hints)


# ============================================================
# Policy Merge
# ============================================================


def merge_effective_policy(
    base: EffectivePolicy,
    extra_rules: Sequence[EngineeringRule],
) -> EffectivePolicy:

    rules: list[EngineeringRule] = []

    seen: set[str] = set()

    for rule in [
        *base.rules,
        *extra_rules,
    ]:

        if rule.rule_id in seen:
            continue

        seen.add(rule.rule_id)

        rules.append(rule)

    return EffectivePolicy(rules=rules)


def strategy_feedback_rule(
    decision: VerificationStrategyDecision,
) -> EngineeringRule:

    details = "\n".join(
        (f"- {item.code}: " f"{item.relative_path}: " f"{item.message}")
        for item in decision.violations
    )

    return EngineeringRule(
        rule_id=("TEST-STRATEGY-CORRECTION-001"),
        category=("verification_strategy"),
        severity="must",
        source="runtime",
        description=(
            "上一份 Test Proposal "
            "被 Verification Strategy Guard "
            "拒绝。"
            "必须保持同一个 PatchPlan / "
            "Change Surface，"
            "只纠正 Proof Strategy，"
            "不能减少 Verification Obligation，"
            "不能扩大文件范围。\n" + details
        ),
    )


# ============================================================
# Bounded Strategy-aware Patch Generation
# ============================================================


class VerificationStrategyPatchGenerator:
    """
    Test Proposal 在 Side Effect 前
    最多允许一次 Strategy Correction。

    correction != verification repair

    因为此时：
        - 尚未创建 Workspace
        - 尚未 Apply
        - 尚未运行 Verification
    """

    def __init__(
        self,
        *,
        repository_root: str | Path,
        model: Any,
        guard: VerificationStrategyGuard | None = None,
        max_corrections: int = 1,
    ) -> None:

        if max_corrections < 0:
            raise ValueError("max_corrections 必须 >= 0")

        self.repository_root = Path(repository_root)

        self.model = model

        self.guard = guard or VerificationStrategyGuard()

        self.max_corrections = max_corrections

    async def generate(
        self,
        *,
        artifact: AnalysisArtifact,
        plan: PatchPlan,
        preparation: ExecutionPreparation,
        effective_policy: EffectivePolicy,
        strategy: VerificationStrategyPlan | None,
    ) -> StrategyAwarePatchGenerationResult:

        active_policy = effective_policy

        for correction_count in range(self.max_corrections + 1):

            proposal = await PatchGenerator(
                repository_root=(self.repository_root),
                model=self.model,
            ).generate(
                artifact=artifact,
                plan=plan,
                preparation=preparation,
                effective_policy=(active_policy),
            )

            decision = self.guard.evaluate(
                strategy=strategy,
                proposal=proposal,
            )

            if decision.allowed:

                return StrategyAwarePatchGenerationResult(
                    proposal=proposal,
                    correction_count=(correction_count),
                )

            if correction_count >= self.max_corrections:
                raise (
                    VerificationStrategyError(
                        "VerificationStrategy" "Violation: " + decision.format_message()
                    )
                )

            active_policy = merge_effective_policy(
                effective_policy,
                [strategy_feedback_rule(decision)],
            )

        raise AssertionError("unreachable")
