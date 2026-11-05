from dataclasses import dataclass

from agent_core.benchmark.context_file_quality import (
    ContextFileOracle,
)
from agent_core.sdd.model import (
    RequirementSpec,
    TechnicalSpec,
    TaskDefinition,
)


@dataclass(frozen=True)
class ContextSuiteCase:
    case_id: str
    raw_requirement: str
    requirement: RequirementSpec
    technical_spec: TechnicalSpec
    task: TaskDefinition
    oracle: ContextFileOracle
    required_symbol_ids: tuple[str, ...]


def make_case(
    *,
    case_id: str,
    title: str,
    requirement: str,
    investigation_questions: list[str],
    required_files: list[str],
    useful_files: list[str],
    irrelevant_files: list[str],
    required_symbols: list[str],
) -> ContextSuiteCase:
    return ContextSuiteCase(
        case_id=case_id,
        raw_requirement=requirement,
        requirement=RequirementSpec(
            title=title,
            summary=requirement,
            goals=[title],
        ),
        technical_spec=TechnicalSpec(
            summary=f"调查现有实现并完成：{title}",
        ),
        task=TaskDefinition(
            task_id=case_id,
            task_type="code_change",
            title=title,
            objective=requirement,
            source_step_order=1,
            responsibilities=[title],
            investigation_questions=investigation_questions,
            expected_outputs=[title],
            verification=["相关功能与现有行为保持正确"],
        ),
        oracle=ContextFileOracle(
            required_files=required_files,
            useful_files=useful_files,
            irrelevant_files=irrelevant_files,
        ),
        required_symbol_ids=tuple(required_symbols),
    )


def build_cases() -> list[ContextSuiteCase]:
    return [
        make_case(
            case_id="ORDER-001",
            title="订单列表状态筛选",
            requirement=(
                "给订单列表增加状态筛选，" "保持订单加载行为，不要修改用户列表"
            ),
            investigation_questions=[
                "订单列表从哪里加载？",
                "订单状态在哪里定义？",
                "订单数据经过哪些 API 或 Mock？",
            ],
            required_files=[
                "src/views/OrdersView.vue",
                "src/api/orders.ts",
                "src/mocks/service.ts",
                "src/types/domain.ts",
            ],
            useful_files=[
                "src/api/requestClient.ts",
            ],
            irrelevant_files=[
                "src/components/UserFilters.vue",
                "src/components/UserTable.vue",
                "src/components/PaginationBar.vue",
            ],
            required_symbols=[
                "src/views/OrdersView.vue::component::OrdersView",
                "src/api/orders.ts::function::listOrders",
                "src/api/orders.ts::variable::useMock",
                "src/mocks/service.ts::function::mockListOrders",
                "src/types/domain.ts::interface::Order",
                "src/api/requestClient.ts::function::request",
            ],
        ),
        make_case(
            case_id="USER-001",
            title="用户列表状态筛选",
            requirement=(
                "改进用户列表状态筛选，" "保持用户加载和分页行为，" "不要修改订单列表"
            ),
            investigation_questions=[
                "用户筛选组件如何触发状态变化？",
                "用户 Store 怎样发起数据请求？",
                "筛选条件如何传递到 API？",
            ],
            required_files=[
                "src/views/UsersView.vue",
                "src/components/UserFilters.vue",
                "src/stores/users.ts",
                "src/api/users.ts",
            ],
            useful_files=[
                "src/types/domain.ts",
                "src/mocks/service.ts",
                "src/components/PaginationBar.vue",
                "src/components/UserTable.vue",
                "src/api/requestClient.ts",
            ],
            irrelevant_files=[
                "src/views/OrdersView.vue",
                "src/api/orders.ts",
            ],
            required_symbols=[
                "src/api/users.ts::function::listUsers",
                "src/mocks/service.ts::function::mockListUsers",
                "src/types/domain.ts::interface::UserQuery",
            ],
        ),
        make_case(
            case_id="API-001",
            title="请求客户端错误处理",
            requirement=(
                "改进 HTTP 请求客户端的错误处理，" "保持订单和用户 API 调用兼容"
            ),
            investigation_questions=[
                "HTTP 请求异常在哪里统一转换？",
                "订单和用户 API 如何调用请求客户端？",
                "ApiError 如何保留状态码和错误信息？",
            ],
            required_files=[
                "src/api/requestClient.ts",
            ],
            useful_files=[
                "src/api/orders.ts",
                "src/api/users.ts",
                "src/types/domain.ts",
            ],
            irrelevant_files=[
                "src/components/UserFilters.vue",
                "src/components/UserTable.vue",
                "src/components/PaginationBar.vue",
            ],
            required_symbols=[
                "src/api/requestClient.ts::function::request",
            ],
        ),
    ]
