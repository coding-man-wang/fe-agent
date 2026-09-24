from pathlib import Path

from agent_core.codebase.model import CodeChunk, RepositoryFileIndex


class CodeChunker:
    def __init__(self, *, max_lines: int = 40, overlap_lines: int = 5):
        if max_lines < 1:
            raise ValueError("max_lines 必须 >= 1")

        if overlap_lines < 0:
            raise ValueError("overlap_lines 必须 >= 0")

        if overlap_lines >= max_lines:
            raise ValueError("overlap_lines 必须小于 max_lines")

        self.max_lines = max_lines
        self.overlap_lines = overlap_lines

    def build_chunks(self, index: RepositoryFileIndex) -> list[CodeChunk]:
        # chunk_id: str = Field(description="Chunk 唯一 ID")
        #
        # relative_path: str = Field(description="来源文件")
        #
        # start_line: int = Field(ge=1, description="起始行")
        #
        # end_line: int = Field(ge=1, description="结束行")
        #
        # content: str = Field(description="Chunk 的代码内容")
        #
        # language: str | None = None
        #
        # category: str = "source"

        root_path = Path(index.root_path)

        chunks: list[CodeChunk] = []

        for file_record in index.files:
            if not file_record.is_indexable:
                continue

            path = root_path / file_record.relative_path

            file_chunks = self._chunk_file(
                path=path,
                relative_path=(file_record.relative_path),
                language=(file_record.language),
                category=(file_record.category),
            )

            chunks.extend(file_chunks)

        return chunks

    def _chunk_file(
        self,
        *,
        path: Path,
        relative_path: str,
        language: str | None,
        category: str,
    ):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeError, OSError):
            return []

        chunks: list[CodeChunk] = []

        step = self.max_lines - self.overlap_lines

        start = 0
        while start < len(lines):
            end = min(len(lines), start + self.max_lines)

            content = "\n".join(lines[start:end])

            chunk_id = f"{relative_path}:{start + 1}-{end}"

            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    relative_path=relative_path,
                    start_line=start + 1,
                    end_line=end,
                    content=content,
                    language=language,
                    category=category,
                )
            )

            if end == len(lines):
                break

            start += step

        return chunks
