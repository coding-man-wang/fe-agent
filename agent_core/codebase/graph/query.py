from __future__ import annotations

from collections import defaultdict, deque

from pydantic import BaseModel, ConfigDict, Field

from agent_core.codebase.graph.model import RepositoryDependencyGraph


class GraphTraversalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str
    depth: int = Field(ge=1)
    via_path: str


class RepositoryDependencyGraphQuery:
    def __init__(
        self,
        graph: RepositoryDependencyGraph,
    ) -> None:
        self.graph = graph

        self.forward: dict[str, set[str]] = defaultdict(set)
        self.reverse: dict[str, set[str]] = defaultdict(set)

        for edge in graph.edges:
            self.forward[edge.source_path].add(edge.target_path)
            self.reverse[edge.target_path].add(edge.source_path)

    def dependencies(
        self,
        relative_path: str,
        *,
        max_depth: int = 1,
    ) -> list[GraphTraversalHit]:
        return self._traverse(
            start_path=relative_path,
            adjacency=self.forward,
            max_depth=max_depth,
        )

    def dependents(
        self,
        relative_path: str,
        *,
        max_depth: int = 1,
    ) -> list[GraphTraversalHit]:
        return self._traverse(
            start_path=relative_path,
            adjacency=self.reverse,
            max_depth=max_depth,
        )

    def _traverse(
        self,
        *,
        start_path: str,
        adjacency: dict[str, set[str]],
        max_depth: int,
    ) -> list[GraphTraversalHit]:
        if max_depth < 1:
            raise ValueError("max_depth 必须 >= 1")

        queue: deque[tuple[str, int]] = deque(
            [(start_path, 0)]
        )

        visited = {start_path}
        hits: list[GraphTraversalHit] = []

        while queue:
            current_path, depth = queue.popleft()

            if depth >= max_depth:
                continue

            next_depth = depth + 1

            for next_path in sorted(
                adjacency.get(current_path, set())
            ):
                if next_path in visited:
                    continue

                visited.add(next_path)

                hits.append(
                    GraphTraversalHit(
                        relative_path=next_path,
                        depth=next_depth,
                        via_path=current_path,
                    )
                )

                queue.append(
                    (
                        next_path,
                        next_depth,
                    )
                )

        return hits
