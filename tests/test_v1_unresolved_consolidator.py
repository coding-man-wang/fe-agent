from agent_core.change_surface.unresolved import UnresolvedQuestionConsolidator


def test_consolidator_merges_near_duplicate_questions():
    consolidator = UnresolvedQuestionConsolidator(max_items=8)
    result = consolidator.consolidate(
        [
            "PaginationBar.vue 的具体 props/事件尚未确认，需要确认是否支持 pageSize。",
            "PaginationBar.vue 的 props 和事件未在证据中确认，需确认 pageSize 接口。",
            "现有请求防抖/去重机制未确认。",
            "现有请求防抖或去重机制未在证据中明确。",
        ]
    )

    assert len(result) <= 2


def test_consolidator_caps_case_level_questions():
    consolidator = UnresolvedQuestionConsolidator(max_items=3)
    result = consolidator.consolidate(
        [f"独立问题 {index} 需要确认" for index in range(10)]
    )

    assert len(result) <= 3
