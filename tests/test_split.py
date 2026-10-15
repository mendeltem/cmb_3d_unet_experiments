import pytest

from cmbx.data import fold_roles, make_split


def _cases(n=30):
    cases = [dict(id=f"c{i:02d}", group="A" if i % 3 else "B") for i in range(n)]
    counts = {c["id"]: (i * 7) % 5 for i, c in enumerate(cases)}
    return cases, counts


def test_split_deterministic_and_complete():
    cases, counts = _cases()
    a = make_split(cases, counts, 5, seed=0)
    b = make_split(cases, counts, 5, seed=0)
    c = make_split(cases, counts, 5, seed=1)
    assert a == b
    assert a["folds"] != c["folds"]
    ids = sorted(x for f in a["folds"] for x in f)
    assert ids == sorted(c["id"] for c in cases)
    assert max(a["cases_per_fold"]) - min(a["cases_per_fold"]) <= 1


def test_split_stratifies_load_classes():
    cases, counts = _cases(50)
    s = make_split(cases, counts, 5, seed=0)
    from cmbx.data import load_class
    for cls in (0, 1, 2):
        per_fold = [sum(1 for x in f if load_class(counts[x]) == cls) for f in s["folds"]]
        assert max(per_fold) - min(per_fold) <= 2


def test_fixed_fold_column_is_used():
    cases = [dict(id=f"c{i}", fold=i % 5) for i in range(10)]
    s = make_split(cases, {c["id"]: 0 for c in cases}, 5)
    assert s["folds"][0] == ["c0", "c5"] and s["rule"].startswith("fixed")


def test_fold_roles():
    s = dict(n_folds=5, folds=[[f"f{k}a", f"f{k}b"] for k in range(5)])
    test, inner, train = fold_roles(s, 4)
    assert test == ["f4a", "f4b"] and inner == ["f0a", "f0b"]
    assert sorted(train) == sorted(f"f{k}{x}" for k in (1, 2, 3) for x in "ab")
    with pytest.raises(ValueError):
        fold_roles(dict(n_folds=2, folds=[["a"], ["b"]]), 0)
