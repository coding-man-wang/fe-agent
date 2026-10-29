from __future__ import annotations

from agent_core.sdd.model import (
    TaskDefinition,
)


class TaskVerificationCommandPolicy:
    """
    根据 Task 生命周期选择验证强度。

    code_change:
        验证 production change
        的结构正确性。

    test:
        Test code 本身也是代码，
        因此既需要静态质量检查，
        也需要执行相关测试。

    verification:
        最终完整验证由外层 Workflow 负责。
    """

    STRUCTURAL_COMMANDS = {
        "frontend-lint",
        "frontend-typecheck",
        "frontend-build",
    }

    TEST_EXECUTION_COMMANDS = {
        "frontend-unit-test",
        "frontend-e2e",
    }

    # ==================================
    # Test Authoring Verification
    #
    # 测试文件虽然不属于 production，
    # 但仍然必须满足：
    #
    # - lint
    # - typecheck
    # - behavioral verification
    #
    # build 暂不在每个 test task 重跑，
    # 由 production/final verification
    # 负责完整构建验证。
    # ==================================

    TEST_AUTHORING_COMMANDS = {
        "frontend-lint",
        "frontend-typecheck",
        "frontend-unit-test",
        "frontend-e2e",
    }

    def for_task(
        self,
        *,
        task: TaskDefinition,
        base_command_ids: list[str],
    ) -> list[str]:

        if task.task_type == "code_change":

            result = [
                command_id
                for command_id in base_command_ids
                if command_id in self.STRUCTURAL_COMMANDS
            ]

        elif task.task_type == "test":

            result = [
                command_id
                for command_id in base_command_ids
                if command_id in self.TEST_AUTHORING_COMMANDS
            ]

        else:

            result = list(base_command_ids)

        if not result:

            raise ValueError("当前 Task 没有可执行的 " "Verification Command")

        return result
