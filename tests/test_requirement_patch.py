from agent_core.sdd.model import (
    RequirementPatch,
    RequirementSpec,
    StringListPatch,
)

from agent_core.sdd.patch import (
    apply_requirement_patch,
)


def test_apply_requirement_patch():
    original = RequirementSpec(
        title="用户列表",
        summary="实现用户列表页面",
        goals=[
            "展示用户列表",
        ],
        non_goals=[
            "不实现用户删除",
        ],
        acceptance_criteria=[
            "页面可以展示用户",
        ],
        constraints=[
            "使用 Vue3",
        ],
        assumptions=[],
        open_questions=[
            "是否需要状态筛选？",
        ],
    )

    patch = RequirementPatch(
        goals=StringListPatch(
            add=[
                "支持用户状态筛选"
            ]
        ),
        acceptance_criteria=StringListPatch(
            add=[
                "支持按启用状态筛选用户"
            ]
        ),
        constraints=StringListPatch(
            add=[
                "用户状态仅包含启用和禁用"
            ]
        ),
        open_questions=StringListPatch(
            remove=[
                "是否需要状态筛选？"
            ]
        ),
    )

    updated = apply_requirement_patch(
        original,
        patch,
    )

    assert (
            "展示用户列表"
            in updated.goals
    )

    assert (
            "支持用户状态筛选"
            in updated.goals
    )

    assert (
            updated.open_questions
            == []
    )

    assert (
            "不实现用户删除"
            in updated.non_goals
    )


def test_patch_does_not_modify_original():
    original = RequirementSpec(
        title="用户列表",
        summary="用户列表页面",
        goals=[
            "展示用户"
        ],
        open_questions=[
            "是否需要分页？"
        ],
    )

    patch = RequirementPatch(
        open_questions=StringListPatch(
            remove=[
                "是否需要分页？"
            ]
        )
    )

    updated = apply_requirement_patch(
        original,
        patch,
    )

    assert original.open_questions == [
        "是否需要分页？"
    ]

    assert updated.open_questions == []


def test_patch_does_not_duplicate():
    original = RequirementSpec(
        title="用户列表",
        summary="用户列表",
        goals=[
            "支持用户名搜索"
        ],
    )

    patch = RequirementPatch(
        goals=StringListPatch(
            add=[
                "支持用户名搜索"
            ]
        )
    )

    updated = apply_requirement_patch(
        original,
        patch,
    )

    assert updated.goals == [
        "支持用户名搜索"
    ]
