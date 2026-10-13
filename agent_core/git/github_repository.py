from dataclasses import dataclass
from urllib.parse import urlparse


class GitHubRepositoryError(ValueError):
    pass


@dataclass(frozen=True)
class GitHubRepository:
    owner: str
    name: str

    @property
    def full_name(
        self,
    ) -> str:
        return f"{self.owner}/" f"{self.name}"


def parse_github_remote_url(
    url: str,
) -> GitHubRepository:

    value = url.strip()

    if not value:
        raise GitHubRepositoryError("Git Remote URL 不能为空")

    # --------------------------
    # SSH
    #
    # git@github.com:owner/repo.git
    # --------------------------

    prefix = "git@github.com:"

    if value.startswith(prefix):

        path = value[len(prefix) :]

        return _parse_repository_path(path)

    # --------------------------
    # HTTPS / SSH URL
    #
    # https://github.com/owner/repo.git
    # ssh://git@github.com/owner/repo.git
    # --------------------------

    parsed = urlparse(value)

    if parsed.hostname != "github.com":
        raise GitHubRepositoryError("当前只支持 github.com Remote")

    path = parsed.path.lstrip("/")

    return _parse_repository_path(path)


def _parse_repository_path(
    path: str,
) -> GitHubRepository:

    normalized = path.strip("/")

    if normalized.endswith(".git"):
        normalized = normalized[:-4]

    parts = normalized.split("/")

    if len(parts) != 2:
        raise GitHubRepositoryError("无法从 Remote URL " "解析 GitHub Repository")

    owner = parts[0].strip()

    name = parts[1].strip()

    if not owner or not name:
        raise GitHubRepositoryError("GitHub Repository " "owner/name 非法")

    return GitHubRepository(
        owner=owner,
        name=name,
    )
