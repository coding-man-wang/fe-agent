import asyncio
import json
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.request import (
    Request,
    urlopen,
)

from agent_core.git.client import (
    GitClient,
)
from agent_core.git.cli_client import (
    CliGitClient,
)
from agent_core.git.github_repository import (
    parse_github_remote_url,
)
from agent_core.git.pull_request_model import (
    PullRequestRequest,
    PullRequestResult,
)


class GitHubPullRequestError(RuntimeError):
    pass


class GitHubApiClient(Protocol):

    async def create_pull_request(
        self,
        *,
        repository: str,
        title: str,
        body: str,
        head: str,
        base: str,
    ) -> dict[str, Any]: ...


class DefaultGitHubApiClient:

    def __init__(
        self,
        *,
        token: str | None = None,
        api_base_url: str = ("https://api.github.com"),
    ) -> None:

        self.token = token or os.environ.get("GITHUB_TOKEN")

        if not self.token:
            raise GitHubPullRequestError("缺少 GITHUB_TOKEN")

        self.api_base_url = api_base_url.rstrip("/")

    async def create_pull_request(
        self,
        *,
        repository: str,
        title: str,
        body: str,
        head: str,
        base: str,
    ) -> dict[str, Any]:

        return await asyncio.to_thread(
            self._create_pull_request_sync,
            repository=repository,
            title=title,
            body=body,
            head=head,
            base=base,
        )

    def _create_pull_request_sync(
        self,
        *,
        repository: str,
        title: str,
        body: str,
        head: str,
        base: str,
    ) -> dict[str, Any]:

        payload = json.dumps(
            {
                "title": title,
                "body": body,
                "head": head,
                "base": base,
            }
        ).encode("utf-8")

        url = f"{self.api_base_url}" f"/repos/{repository}/pulls"

        request = Request(
            url=url,
            data=payload,
            method="POST",
            headers={
                "Accept": ("application/vnd.github+json"),
                "Authorization": (f"Bearer {self.token}"),
                "X-GitHub-Api-Version": ("2022-11-28"),
                "Content-Type": ("application/json"),
            },
        )

        try:

            with urlopen(
                request,
                timeout=30,
            ) as response:

                content = response.read()

        except HTTPError as exc:

            try:
                detail = exc.read().decode(
                    "utf-8",
                    errors="replace",
                )

            except Exception:
                detail = ""

            raise GitHubPullRequestError(
                "GitHub 创建 Pull Request " f"失败: HTTP {exc.code} " f"{detail}"
            ) from exc

        try:
            result = json.loads(content.decode("utf-8"))

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:

            raise GitHubPullRequestError("GitHub 返回了无效响应") from exc

        if not isinstance(
            result,
            dict,
        ):
            raise GitHubPullRequestError("GitHub 返回结构非法")

        return result


class GitHubPullRequestProvider:

    def __init__(
        self,
        *,
        repository_root: str | Path,
        git_client: GitClient | None = None,
        api_client: GitHubApiClient | None = None,
    ) -> None:

        self.repository_root = Path(repository_root).resolve()

        if not self.repository_root.is_dir():
            raise GitHubPullRequestError("repository_root 不存在")

        self.git_client = git_client or CliGitClient()

        self.api_client = api_client or DefaultGitHubApiClient()

    async def create(
        self,
        *,
        request: PullRequestRequest,
    ) -> PullRequestResult:

        # --------------------------
        # 1. 获取对应 Remote URL
        # --------------------------

        remote_result = await self.git_client.run(
            cwd=self.repository_root,
            args=[
                "remote",
                "get-url",
                request.remote_name,
            ],
        )

        remote_url = remote_result.stdout.strip()

        repository = parse_github_remote_url(remote_url)

        # --------------------------
        # 2. 调 GitHub API
        # --------------------------

        response = await self.api_client.create_pull_request(
            repository=(repository.full_name),
            title=request.title,
            body=request.body,
            head=(request.head_branch),
            base=(request.base_branch),
        )

        # --------------------------
        # 3. 校验响应
        # --------------------------

        number = response.get("number")

        html_url = response.get("html_url")

        head = response.get("head")

        base = response.get("base")

        if (
            not isinstance(
                number,
                int,
            )
            or number < 1
        ):
            raise GitHubPullRequestError("GitHub PR number 非法")

        if (
            not isinstance(
                html_url,
                str,
            )
            or not html_url
        ):
            raise GitHubPullRequestError("GitHub PR URL 非法")

        if not isinstance(
            head,
            dict,
        ):
            raise GitHubPullRequestError("GitHub PR head 非法")

        if not isinstance(
            base,
            dict,
        ):
            raise GitHubPullRequestError("GitHub PR base 非法")

        head_ref = head.get("ref")

        base_ref = base.get("ref")

        head_sha = head.get("sha")

        if head_ref != request.head_branch:
            raise GitHubPullRequestError("GitHub PR Head Branch " "与请求不一致")

        if base_ref != request.base_branch:
            raise GitHubPullRequestError("GitHub PR Base Branch " "与请求不一致")

        if head_sha != request.commit_sha:
            raise GitHubPullRequestError("GitHub PR Head Commit " "与请求不一致")

        return PullRequestResult(
            handoff_id=(request.handoff_id),
            task_id=(request.task_id),
            proposal_id=(request.proposal_id),
            provider="github",
            repository=(repository.full_name),
            number=number,
            url=html_url,
            head_branch=(request.head_branch),
            base_branch=(request.base_branch),
            commit_sha=(request.commit_sha),
        )
