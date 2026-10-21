import pytest

from agent_core.benchmark.text_model import BenchmarkTextModelError, extract_json_object


def test_extract_json_object_accepts_markdown_fence():
    assert extract_json_object('```json\n{"ok": true}\n```') == '{"ok": true}'


def test_extract_json_object_accepts_surrounding_explanation():
    assert extract_json_object('result:\n{"ok": true}\nend') == '{"ok": true}'


def test_extract_json_object_rejects_non_json():
    with pytest.raises(BenchmarkTextModelError):
        extract_json_object('nothing useful')
