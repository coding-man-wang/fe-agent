from __future__ import annotations

from agent_core.sdd.model import (
    TaskDefinition,
)


class TaskVerificationCommandPolicy:
    """
    根据 Task 生命周期选择验证强度。

    code_change:
        先做结构验证。

    test:
        测试文件更新之后执行 unit test。

    verification:
        最终完整验证由外层 Workflow 负责。
    """

    STRUCTURAL_COMMANDS = {
        "frontend-lint",
        "frontend-typecheck",
        "frontend-build",
    }

    TEST_COMMANDS = {
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
                if command_id in self.TEST_COMMANDS
            ]

        else:
            result = list(base_command_ids)

        if not result:
            raise ValueError("当前 Task 没有可执行的 " "Verification Command")

        return result
