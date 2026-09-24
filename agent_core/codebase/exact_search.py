from pathlib import Path

from agent_core.codebase.model import (
    RepositoryFileIndex,
    ExactSearchResponse,
    ExactSearchResult,
    ExactSearchMatch,
)


class ExactCodeSearcher:
    def __init__(
        self, *, max_content_matches_per_file: int = 5, snippet_radius: int = 80
    ):
        self.max_content_matches_per_file = max_content_matches_per_file
        self.snippet_radius = snippet_radius

    def search(
        self,
        index: RepositoryFileIndex,
        query: str,
        *,
        case_sensitive: bool = False,
        max_result: int = 20,
    ) -> ExactSearchResponse:
        if not query.strip():
            raise ValueError("搜索关键词不能为空")

        normalized_query = query if case_sensitive else query.lower()

        root = Path(index.root_path)

        results: list[ExactSearchResult] = []

        scanned_files = 0
        for file_record in index.files:
            scanned_files += 1

            matches: list[ExactSearchMatch] = []

            score = 0

            filename = file_record.name if case_sensitive else file_record.name.lower()

            relative_path = (
                file_record.relative_path
                if case_sensitive
                else file_record.relative_path.lower()
            )

            # 1. 文件名匹配
            if filename == normalized_query:
                score += 100

                matches.append(
                    ExactSearchMatch(
                        match_type="filename",
                        snippet=file_record.name,
                    )
                )
            elif filename in normalized_query:
                score += 60
                matches.append(
                    ExactSearchMatch(
                        match_type="filename",
                        snippet=file_record.name,
                    )
                )

            # 2. 路径匹配
            if normalized_query in relative_path:
                score += 40

                matches.append(
                    ExactSearchMatch(
                        match_type="path",
                        snippet=(file_record.relative_path),
                    )
                )

            if file_record.is_indexable:
                content_matches = self._search_file_content(
                    root=root,
                    relative_path=(file_record.relative_path),
                    query=query,
                    case_sensitive=case_sensitive,
                )

                matches.extend(content_matches)

                score += len(content_matches) * 10

            if not matches:
                continue

            results.append(
                ExactSearchResult(
                    relative_path=file_record.relative_path,
                    category=file_record.category,
                    language=file_record.language,
                    score=score,
                    matches=matches,
                )
            )

        results.sort(key=lambda result: (-result.score, result.relative_path))

        limited_results = results[:max_result]

        return ExactSearchResponse(
            query=query,
            scanned_files=scanned_files,
            matched_files=len(results),
            results=limited_results,
        )

    def _search_file_content(
        self, *, root: Path, relative_path: str, query: str, case_sensitive: bool
    ):
        path = root / relative_path
        matches: list[ExactSearchMatch] = []

        try:
            with path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    searchable_line = line if case_sensitive else line.lower()

                    searchable_query = query if case_sensitive else query.lower()

                    match_index = searchable_line.find(searchable_query)

                    if match_index == -1:
                        continue

                    snippet = self._build_snippet(
                        line=line,
                        match_index=match_index,
                        query_length=len(query),
                    )

                    matches.append(
                        ExactSearchMatch(
                            match_type="content",
                            line_number=line_number,
                            snippet=snippet,
                        )
                    )

                    if len(matches) >= self.max_content_matches_per_file:
                        break

        except (
            UnicodeDecodeError,
            OSError,
        ):
            return []

        return matches

    def _build_snippet(self, *, line: str, match_index: int, query_length: int) -> str:
        start = max(0, match_index - self.snippet_radius)

        end = min(len(line), match_index + self.snippet_radius + query_length)

        return line[start:end].strip()
