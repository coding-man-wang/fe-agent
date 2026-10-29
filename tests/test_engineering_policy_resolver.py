from agent_core.policies.resolver import (
    EngineeringPolicyResolver,
)
from agent_core.sdd.model import (
    TaskDefinition,
)


def build_task(
    task_type: str,
) -> TaskDefinition:

    return TaskDefinition(
        task_id="TASK-001",
        task_type=task_type,
        title="test",
        objective="test",
        source_step_order=1,
        depends_on=[],
        inputs=[],
        target_modules=[],
        expected_outputs=[],
        verification=[],
        acceptance_criteria_refs=[],
    )


def test_test_task_gets_default_test_policy():

    policy = EngineeringPolicyResolver().resolve(
        task=build_task("test"),
        project_rules=[],
    )

    ids = set(policy.rule_ids())

    assert "TEST-BEHAVIOR-001" in ids

    assert "TEST-ENV-001" in ids


def test_code_change_does_not_get_test_policy():

    policy = EngineeringPolicyResolver().resolve(
        task=build_task("code_change"),
        project_rules=[],
    )

    assert not any(rule.rule_id.startswith("TEST-") for rule in policy.rules)


def test_project_rules_are_preserved():

    policy = EngineeringPolicyResolver().resolve(
        task=build_task("test"),
        project_rules=["E2E selector " "优先使用 role/name"],
    )

    project_rules = [rule for rule in policy.rules if (rule.source == "project")]

    assert len(project_rules) == 1

    assert project_rules[0].description == ("E2E selector " "优先使用 role/name")


def test_project_rule_id_is_stable():

    resolver = EngineeringPolicyResolver()

    first = resolver.resolve(
        task=build_task("test"),
        project_rules=["禁止使用 CSS selector"],
    )

    second = resolver.resolve(
        task=build_task("test"),
        project_rules=["禁止使用 CSS selector"],
    )

    first_id = [item.rule_id for item in first.rules if item.source == "project"][0]

    second_id = [item.rule_id for item in second.rules if item.source == "project"][0]

    assert first_id == second_id
