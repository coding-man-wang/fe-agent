from agent_core.coding.verification_command_model import (
    AllowedVerificationCommand,
)


class VerificationPolicyError(ValueError):
    pass


class VerificationCommandPolicy:
    """
    验证命令白名单。

    Agent 只能通过 command_id
    选择预定义命令。
    """

    def __init__(
        self,
        commands: list[AllowedVerificationCommand],
    ) -> None:

        if not commands:
            raise VerificationPolicyError("Verification Command Policy 不能为空")

        command_map: dict[
            str,
            AllowedVerificationCommand,
        ] = {}

        for command in commands:

            if command.command_id in command_map:
                raise VerificationPolicyError(
                    "重复 command_id: " f"{command.command_id}"
                )

            command_map[command.command_id] = command

        self._commands = command_map

    def resolve(
        self,
        command_id: str,
    ) -> AllowedVerificationCommand:

        command = self._commands.get(command_id)

        if command is None:
            raise VerificationPolicyError(
                "未授权的 Verification Command: " f"{command_id}"
            )

        return command.model_copy(deep=True)

    def list_command_ids(
        self,
    ) -> list[str]:

        return sorted(self._commands.keys())
