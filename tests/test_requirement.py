from agent_core.sdd.model import RequirementSpec


def test_requirement_spec():
    data = {
        "title": "用户列表",
        "summary": "新增用户列表页面",
        "goals": [
            "展示用户列表",
            "支持搜索",
        ],
        "non_goals": [
            "不实现用户删除",
        ],
        "acceptance_criteria": [
            "页面能够展示用户",
        ],
        "constraints": [
            "使用 Vue3",
        ],
        "assumptions": [
            "用户列表 API 已存在",
        ],
        "open_questions": [
            "是否需要状态筛选？",
        ],
    }

    spec = RequirementSpec.model_validate(data)

    assert spec.title == "用户列表"

    assert len(spec.goals) == 2

    assert spec.constraints[0] == "使用 Vue3"


def test_requirement_from_json():
    content = """
    {
        "title": "用户列表",
        "summary": "新增用户列表页面",
        "goals": ["展示用户列表"],
        "non_goals": [],
        "acceptance_criteria": [
            "页面能够正常展示数据"
        ],
        "constraints": ["Vue3"],
        "assumptions": [],
        "open_questions": []
    }
    """

    spec = RequirementSpec.model_validate_json(content)

    assert spec.title == "用户列表"
