from agent_core.sdd.model import RequirementSpec, RequirementPatch
from agent_core.sdd.model import StringListPatch


def apply_list_patch(current: list[str], patch: StringListPatch) -> list[str]:
    remove_items = set(patch.remove)

    result = [
        item for item in current if item not in remove_items
    ]

    for item in patch.add:
        if item not in result:
            result.append(item)

    return result


def apply_requirement_patch(
        spec: RequirementSpec,
        patch: RequirementPatch
) -> RequirementSpec:
    updated = spec.model_copy(deep=True)

    if patch.title is not None:
        updated.title = patch.title

    if patch.summary is not None:
        updated.summary = patch.summary

    updated.goals = apply_list_patch(updated.goals, patch.goals)

    updated.non_goals = apply_list_patch(
        updated.non_goals,
        patch.non_goals
    )

    updated.acceptance_criteria = apply_list_patch(
        updated.acceptance_criteria,
        patch.acceptance_criteria
    )

    updated.constraints = apply_list_patch(
        updated.constraints,
        patch.constraints
    )

    updated.assumptions = apply_list_patch(
        updated.assumptions,
        patch.assumptions
    )

    updated.open_questions = apply_list_patch(
        updated.open_questions,
        patch.open_questions
    )

    return updated
