from experiments.answer_safety_suite import _select


def test_select_is_deterministic_disjoint_and_marks_role():
    pool = [{"id": f"case-{number}"} for number in range(6)]
    used = {"case-0"}

    first = _select(role="numeric", pool=pool, count=3, used=used)
    second_used = {"case-0"}
    second = _select(role="numeric", pool=pool, count=3, used=second_used)

    assert first == second
    assert all(case["answer_suite_role"] == "numeric" for case in first)
    assert "case-0" not in {case["id"] for case in first}
    assert used == {"case-0", *(case["id"] for case in first)}
