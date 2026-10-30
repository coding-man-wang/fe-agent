from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from agent_core.change_surface.model import (
    ResolvedChangeSurface,
)
from agent_core.sdd.model import (
    TaskDefinition,
    TaskPlan,
    ExecutionObligation,
)
from agent_core.sdd.task_graph import (
    topological_sort_tasks,
)

# ==========================================
# Execution Plan Compression Model
# ==========================================


class ExecutionTaskGroup(BaseModel):
    """
    一组被聚合成单个 Runtime Execution Unit
    的 Logical code_change Tasks。

    leader_task_id:
        真正进入 FullExecution 的 Task ID。

    member_task_ids:
        所有被该 Execution Unit 覆盖的
        Logical Task。

        第一项固定为 leader。
    """

    model_config = ConfigDict(extra="forbid")

    leader_task_id: str

    member_task_ids: list[str] = Field(
        default_factory=list,
    )

    baseline_target_files: list[str] = Field(
        default_factory=list,
    )

    reasons: list[str] = Field(
        default_factory=list,
    )


class ExecutionPlanCompressionSummary(BaseModel):
    """
    写入 FullExecutionRunRecord 的
    轻量 Observability 数据。

    不保存整个 Execution TaskPlan，
    避免 Run Record 重复存储大量内容。
    """

    model_config = ConfigDict(extra="forbid")

    logical_task_count: int = Field(ge=0)

    execution_task_count: int = Field(ge=0)

    coalesced_task_count: int = Field(ge=0)

    groups: list[ExecutionTaskGroup] = Field(default_factory=list)


class ExecutionPlanCompressionResult(BaseModel):
    """
    ExecutionPlanCompressor 的完整结果。

    execution_task_plan:
        真正交给 FullExecution 的计划。

    leader_by_task_id:
        Logical Task → Execution Leader。

        没有被合并的 Task：
            TASK-X -> TASK-X

        被合并的 Task：
            TASK-005 -> TASK-004
    """

    model_config = ConfigDict(extra="forbid")

    execution_task_plan: TaskPlan

    leader_by_task_id: dict[
        str,
        str,
    ] = Field(default_factory=dict)

    summary: ExecutionPlanCompressionSummary

    def group_for_leader(
        self,
        task_id: str,
    ) -> ExecutionTaskGroup | None:

        for group in self.summary.groups:

            if group.leader_task_id == task_id:
                return group

        return None

    def coalesced_task_ids_for_leader(
        self,
        task_id: str,
    ) -> list[str]:

        group = self.group_for_leader(task_id)

        if group is None:
            return []

        return list(group.member_task_ids[1:])


# ==========================================
# Execution Plan Compressor
# ==========================================


class ExecutionPlanCompressor:
    """
    把细粒度 Logical TaskPlan
    转换成较粗粒度 Runtime Execution Plan。

    核心原则：

        Planning Granularity
        !=
        Execution Granularity

    当前版本只允许压缩：

        code_change
            +
        code_change

    永远不合并：

        investigation
        test
        verification

    Compression 使用的是：

        Prepared / Baseline
        ResolvedChangeSurface

    这些 Surface 只用于：

        判断哪些 Logical Tasks
        可以属于同一个 Execution Unit。

    它们绝不会成为最终 Write Permission。

    FullExecution 仍然必须：

        current revision
            ↓
        fresh context
            ↓
        fresh change surface
            ↓
        execution surface
            ↓
        side effect
    """

    MERGEABLE_TASK_TYPE = "code_change"

    LIFECYCLE_BARRIERS = {
        "test",
        "verification",
    }

    # ------------------------------
    # TaskDefinition 中需要合并的
    # list[str] 字段。
    #
    # 使用存在性判断是为了：
    #
    # - 兼容 legacy TaskDefinition
    # - 兼容 v1.3 Logical TaskDefinition
    #
    # 当前 v1.3 会命中：
    #
    # responsibilities
    # investigation_questions
    # repository_hints
    #
    # legacy 仓库则自动忽略。
    # ------------------------------

    MERGE_LIST_FIELDS = (
        "inputs",
        "responsibilities",
        "investigation_questions",
        "repository_hints",
        "target_modules",
        "expected_outputs",
        "verification",
        "acceptance_criteria_refs",
    )

    def compress(
        self,
        *,
        task_plan: TaskPlan,
        task_surfaces: dict[
            str,
            ResolvedChangeSurface,
        ],
    ) -> ExecutionPlanCompressionResult:

        ordered_tasks = topological_sort_tasks(task_plan)

        task_map = {task.task_id: task for task in task_plan.tasks}

        ancestor_map = {
            task.task_id: self._ancestor_task_ids(
                task=task,
                task_map=task_map,
            )
            for task in task_plan.tasks
        }

        # 默认每个 Task 都映射到自己。
        #
        # 只有真正成功进入 Group 的
        # follower 才改成 leader。
        leader_by_task_id = {task.task_id: task.task_id for task in task_plan.tasks}

        follower_ids: set[str] = set()

        groups: list[ExecutionTaskGroup] = []

        # ==================================
        # Phase 1
        #
        # Discover Compression Groups
        # ==================================

        for (
            leader_index,
            leader,
        ) in enumerate(ordered_tasks):

            if leader.task_type != self.MERGEABLE_TASK_TYPE:
                continue

            if leader.task_id in follower_ids:
                continue

            leader_surface = task_surfaces.get(leader.task_id)

            if not self._surface_is_eligible(
                task=leader,
                surface=leader_surface,
            ):
                continue

            assert leader_surface is not None

            leader_targets = self._target_set(leader_surface)

            leader_acceptance = set(leader.acceptance_criteria_refs)

            # 没有 Requirement Contract
            # 交集证据时，不做激进合并。
            if not leader_acceptance:
                continue

            leader_ancestors = ancestor_map[leader.task_id]

            members: list[TaskDefinition] = [leader]

            member_ids: set[str] = {leader.task_id}

            for candidate in ordered_tasks[leader_index + 1 :]:

                if candidate.task_type != self.MERGEABLE_TASK_TYPE:
                    continue

                if candidate.task_id in follower_ids:
                    continue

                candidate_ancestors = ancestor_map[candidate.task_id]

                # --------------------------
                # Safety Rule 1
                #
                # Follower 必须真的是
                # Leader 的 downstream。
                #
                # 不能因为碰巧修改相同文件
                # 就合并独立任务。
                # --------------------------

                if leader.task_id not in candidate_ancestors:
                    continue

                candidate_surface = task_surfaces.get(candidate.task_id)

                if not (
                    self._surface_is_eligible(
                        task=candidate,
                        surface=(candidate_surface),
                    )
                ):
                    continue

                assert candidate_surface is not None

                candidate_targets = self._target_set(candidate_surface)

                # --------------------------
                # Safety Rule 2
                #
                # Leader 必须是
                # Baseline Write Surface
                # 的 dominant owner。
                #
                # follower 不能引入
                # leader 原本完全不知道的
                # production target。
                # --------------------------

                if not (candidate_targets <= leader_targets):
                    continue

                # --------------------------
                # Safety Rule 3
                #
                # 两个任务必须共享至少一个
                # Requirement Acceptance
                # Criterion。
                #
                # 避免：
                #
                # “碰巧修改同一个文件”
                # 被错误解释为
                # “同一个行为契约”。
                # --------------------------

                candidate_acceptance = set(candidate.acceptance_criteria_refs)

                if not (leader_acceptance & candidate_acceptance):
                    continue

                # --------------------------
                # Safety Rule 4
                #
                # 不能跨：
                #
                # test
                # verification
                #
                # Lifecycle Boundary。
                # --------------------------

                if self._has_lifecycle_barrier(
                    ancestor_ids=(candidate_ancestors),
                    task_map=task_map,
                ):
                    continue

                # --------------------------
                # Safety Rule 5
                #
                # Candidate 的 code_change
                # ancestors：
                #
                # 要么已经是 Leader upstream，
                # 要么已经被当前 Group 吸收。
                #
                # 防止压缩后构造：
                #
                # A depends B
                # B depends A
                #
                # 这种人工 Cycle。
                # --------------------------

                if not (
                    self._code_ancestors_compatible(
                        candidate_ancestor_ids=(candidate_ancestors),
                        leader_ancestor_ids=(leader_ancestors),
                        current_member_ids=(member_ids),
                        task_map=task_map,
                    )
                ):
                    continue

                members.append(candidate)

                member_ids.add(candidate.task_id)

                follower_ids.add(candidate.task_id)

                leader_by_task_id[candidate.task_id] = leader.task_id

            if len(members) <= 1:
                continue

            groups.append(
                ExecutionTaskGroup(
                    leader_task_id=(leader.task_id),
                    member_task_ids=[item.task_id for item in members],
                    baseline_target_files=(
                        self._stable_unique(list(leader_surface.implementation_targets))
                    ),
                    reasons=[
                        ("仅聚合 code_change " "生命周期任务"),
                        ("所有 follower 都是 " "leader 的 DAG downstream"),
                        (
                            "所有 follower 的 baseline "
                            "implementation_targets "
                            "均被 leader 完整覆盖"
                        ),
                        (
                            "所有 follower 与 leader "
                            "至少共享一个 Requirement "
                            "Acceptance Criterion"
                        ),
                        (
                            "最终 Write Permission "
                            "仍由执行时 Fresh "
                            "Change Surface 决定"
                        ),
                    ],
                )
            )

        # ==================================
        # Phase 2
        #
        # Build Execution TaskPlan
        # ==================================

        group_by_leader = {group.leader_task_id: group for group in groups}

        execution_tasks: list[TaskDefinition] = []

        for task in ordered_tasks:

            leader_id = leader_by_task_id[task.task_id]

            # Follower 不进入
            # Runtime Execution DAG。
            if leader_id != task.task_id:
                continue

            group = group_by_leader.get(task.task_id)

            if group is None:

                execution_task = task.model_copy(deep=True)

            else:

                member_tasks = [task_map[task_id] for task_id in group.member_task_ids]

                execution_task = self._merge_group(
                    leader=task,
                    members=member_tasks,
                )

            rewritten_dependencies = self._rewrite_dependencies(
                dependency_ids=(execution_task.depends_on),
                leader_by_task_id=(leader_by_task_id),
                own_task_id=(execution_task.task_id),
            )

            execution_task = execution_task.model_copy(
                deep=True,
                update={"depends_on": (rewritten_dependencies)},
            )

            execution_tasks.append(execution_task)

        execution_task_plan = task_plan.model_copy(
            deep=True,
            update={
                "tasks": execution_tasks,
            },
        )

        # ==================================
        # Deterministic Integrity Check
        #
        # Compression 本身不能制造 Cycle。
        # ==================================

        topological_sort_tasks(execution_task_plan)

        self._validate_compression(
            logical_plan=task_plan,
            execution_plan=(execution_task_plan),
            leader_by_task_id=(leader_by_task_id),
            groups=groups,
        )

        summary = ExecutionPlanCompressionSummary(
            logical_task_count=(len(task_plan.tasks)),
            execution_task_count=(len(execution_task_plan.tasks)),
            coalesced_task_count=(
                len(task_plan.tasks) - len(execution_task_plan.tasks)
            ),
            groups=groups,
        )

        return ExecutionPlanCompressionResult(
            execution_task_plan=(execution_task_plan),
            leader_by_task_id=(leader_by_task_id),
            summary=summary,
        )

    # ==================================
    # Eligibility
    # ==================================

    @staticmethod
    def _surface_is_eligible(
        *,
        task: TaskDefinition,
        surface: ResolvedChangeSurface | None,
    ) -> bool:

        if task.task_type != "code_change":
            return False

        if surface is None:
            return False

        if surface.task_id != task.task_id:
            return False

        if surface.execution_disposition != "change_required":
            return False

        if not (surface.implementation_targets):
            return False

        # 当前 Production code_change
        # 不允许通过 Compression
        # 顺便吸收 prospective create。
        #
        # Test Create Authoring
        # 仍走独立 test lifecycle。
        if surface.create_files:
            return False

        return True

    # ==================================
    # DAG Helpers
    # ==================================

    @staticmethod
    def _ancestor_task_ids(
        *,
        task: TaskDefinition,
        task_map: dict[
            str,
            TaskDefinition,
        ],
    ) -> set[str]:

        visited: set[str] = set()

        stack = list(task.depends_on)

        while stack:

            task_id = stack.pop()

            if task_id in visited:
                continue

            upstream = task_map.get(task_id)

            if upstream is None:
                continue

            visited.add(task_id)

            stack.extend(upstream.depends_on)

        return visited

    def _has_lifecycle_barrier(
        self,
        *,
        ancestor_ids: set[str],
        task_map: dict[
            str,
            TaskDefinition,
        ],
    ) -> bool:

        for task_id in ancestor_ids:

            task = task_map.get(task_id)

            if task is None:
                continue

            if task.task_type in self.LIFECYCLE_BARRIERS:
                return True

        return False

    @staticmethod
    def _code_ancestors_compatible(
        *,
        candidate_ancestor_ids: set[str],
        leader_ancestor_ids: set[str],
        current_member_ids: set[str],
        task_map: dict[
            str,
            TaskDefinition,
        ],
    ) -> bool:

        for ancestor_id in candidate_ancestor_ids:

            ancestor = task_map.get(ancestor_id)

            if ancestor is None:
                continue

            if ancestor.task_type != "code_change":
                continue

            # 已经是 Leader 自己的上游，
            # 不会因为 Compression
            # 新增依赖。
            if ancestor_id in leader_ancestor_ids:
                continue

            # 已经被本 Group 吸收。
            if ancestor_id in current_member_ids:
                continue

            return False

        return True

    # ==================================
    # Merge
    # ==================================

    def _merge_group(
        self,
        *,
        leader: TaskDefinition,
        members: list[TaskDefinition],
    ) -> TaskDefinition:

        member_ids = {item.task_id for item in members}

        external_dependencies = self._stable_unique(
            [
                dependency
                for member in members
                for dependency in member.depends_on
                if dependency not in member_ids
            ]
        )

        objectives = self._stable_unique(
            [member.objective for member in members if member.objective.strip()]
        )

        if len(objectives) == 1:
            merged_objective = objectives[0]

        else:
            merged_objective = (
                "当前 Execution Unit "
                "需要一次性满足以下 "
                "Logical Task 目标：\n"
                + "\n".join(
                    (f"- [{member.task_id}] " f"{member.objective}")
                    for member in members
                )
            )

        # ==================================
        # Compression Obligation
        #
        # 无论 Logical Tasks 的 objective
        # 是否相同，都必须逐项保留
        # Execution Obligation。
        #
        # Execution Granularity 可以压缩，
        # Execution Obligation 不能丢失。
        # ==================================

        execution_obligations = [
            ExecutionObligation(
                obligation_id=(member.task_id),
                source_task_id=(member.task_id),
                objective=(member.objective),
                responsibilities=list(
                    getattr(
                        member,
                        "responsibilities",
                        [],
                    )
                ),
                expected_outputs=list(member.expected_outputs),
                acceptance_criteria_refs=list(member.acceptance_criteria_refs),
            )
            for member in members
        ]

        updates: dict = {
            "objective": merged_objective,
            "depends_on": external_dependencies,
            "execution_obligations": execution_obligations,
        }

        available_fields = type(leader).model_fields

        for field_name in self.MERGE_LIST_FIELDS:

            if field_name not in available_fields:
                continue

            merged_values = self._stable_unique(
                [
                    value
                    for member in members
                    for value in getattr(
                        member,
                        field_name,
                        [],
                    )
                ]
            )

            updates[field_name] = merged_values

        return leader.model_copy(
            deep=True,
            update=updates,
        )

    # ==================================
    # Dependency Rewrite
    # ==================================

    def _rewrite_dependencies(
        self,
        *,
        dependency_ids: list[str],
        leader_by_task_id: dict[
            str,
            str,
        ],
        own_task_id: str,
    ) -> list[str]:

        rewritten: list[str] = []

        for dependency_id in dependency_ids:

            mapped_id = leader_by_task_id.get(
                dependency_id,
                dependency_id,
            )

            if mapped_id == own_task_id:
                continue

            rewritten.append(mapped_id)

        return self._stable_unique(rewritten)

    # ==================================
    # Integrity
    # ==================================

    @staticmethod
    def _validate_compression(
        *,
        logical_plan: TaskPlan,
        execution_plan: TaskPlan,
        leader_by_task_id: dict[
            str,
            str,
        ],
        groups: list[ExecutionTaskGroup],
    ) -> None:

        logical_map = {task.task_id: task for task in logical_plan.tasks}

        execution_map = {task.task_id: task for task in execution_plan.tasks}

        # ------------------------------
        # Non-code lifecycle tasks
        # 必须完整保留。
        # ------------------------------

        for task in logical_plan.tasks:

            if task.task_type == "code_change":
                continue

            if task.task_id not in execution_map:
                raise RuntimeError(
                    "Execution Plan Compression "
                    "非法删除了非 code_change "
                    f"Task: {task.task_id}"
                )

            if leader_by_task_id[task.task_id] != task.task_id:
                raise RuntimeError(
                    "Execution Plan Compression "
                    "非法合并了非 code_change "
                    f"Task: {task.task_id}"
                )

        # ------------------------------
        # 每个 follower 必须指向
        # 一个真正存在的 code_change
        # execution leader。
        # ------------------------------

        for (
            logical_task_id,
            leader_id,
        ) in leader_by_task_id.items():

            if logical_task_id == leader_id:
                continue

            leader = execution_map.get(leader_id)

            follower = logical_map.get(logical_task_id)

            if leader is None or follower is None:
                raise RuntimeError("Execution Plan Compression " "产生无效 Task 映射")

            if leader.task_type != "code_change" or follower.task_type != "code_change":
                raise RuntimeError(
                    "Execution Plan Compression " "只允许聚合 code_change"
                )

        # ------------------------------
        # Group 自身必须一致。
        # ------------------------------

        seen_members: set[str] = set()

        for group in groups:

            if not group.member_task_ids:
                raise RuntimeError("Execution Task Group " "不能为空")

            if group.member_task_ids[0] != group.leader_task_id:
                raise RuntimeError("Execution Task Group " "第一项必须是 leader")

            for task_id in group.member_task_ids:

                if task_id in seen_members:
                    raise RuntimeError(
                        "Logical Task "
                        "不能属于多个 "
                        "Execution Groups: "
                        f"{task_id}"
                    )

                seen_members.add(task_id)

    # ==================================
    # Common Helpers
    # ==================================

    @staticmethod
    def _target_set(
        surface: ResolvedChangeSurface,
    ) -> set[str]:

        return {
            path.replace("\\", "/").strip("/")
            for path in (surface.implementation_targets)
            if path.strip()
        }

    @staticmethod
    def _stable_unique(
        values: list[str],
    ) -> list[str]:

        result: list[str] = []

        seen: set[str] = set()

        for value in values:

            if not value:
                continue

            if value in seen:
                continue

            seen.add(value)

            result.append(value)

        return result
