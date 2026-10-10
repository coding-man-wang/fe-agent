class GitRepositoryError(RuntimeError):
    """Git Repository 操作失败。"""


class NotGitRepositoryError(GitRepositoryError):
    """目标路径不是有效 Git Repository。"""


class GitCommandError(GitRepositoryError):
    """
    Git 命令执行失败。
    """

    def __init__(
            self,
            *,
            args: list[str],
            exit_code: int,
            stderr: str,
    ) -> None:
        self.args = list(args)
        self.exit_code = exit_code
        self.stderr = stderr

        super().__init__(
            "Git command failed: "
            f"git {' '.join(args)}, "
            f"exit_code={exit_code}, "
            f"stderr={stderr.strip()}"
        )


class GitWorkspaceError(GitRepositoryError):
    """Agent Git Workspace 操作失败。"""


class GitWorkspaceConflictError(GitWorkspaceError):
    """
    Workspace 当前状态与请求不一致，
    禁止自动覆盖。
    """
