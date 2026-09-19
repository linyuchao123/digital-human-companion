import json
from pathlib import Path


CASES = Path(__file__).parents[1] / "eval" / "emotional_dialogue" / "cases.json"


def test_emotional_evaluation_has_24_handwritten_single_and_multi_turn_cases():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    assert len(cases) >= 24
    assert len({case["id"] for case in cases}) == len(cases)
    assert any(len(case["turns"]) == 1 for case in cases)
    assert any(len(case["turns"]) > 1 for case in cases)
    for case in cases:
        assert case["turns"] and all(isinstance(turn, str) and turn.strip() for turn in case["turns"])
        assert len(case["checks"]) >= 2


def test_evaluation_includes_critical_boundaries_and_user_correction():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    text = json.dumps(cases, ensure_ascii=False)
    for boundary in ("危机分流", "药物", "排他依赖", "恋爱关系", "用户感受优先", "接受更正"):
        assert boundary in text
