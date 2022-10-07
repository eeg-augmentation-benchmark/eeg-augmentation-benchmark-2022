import pytest

from BAE.scoring import compute_relative_improvement
from BAE.scoring import compute_score


@pytest.mark.parametrize("per_fold", [True, False])
@pytest.mark.parametrize("per_class", [None, ['1', '2']])
def test_compute_score(per_fold, per_class, dummy_prediction):
    score = compute_score(
        dummy_prediction,
        per_fold=per_fold,
        per_class=per_class)
    if per_class:
        assert isinstance(score['f1_score'][1], float)
    elif per_fold:
        assert isinstance(score['score'][0], float)
    else:
        assert isinstance(score['avg_score'][0], float)


@pytest.mark.parametrize("per_class", [None, ['1', '2']])
def test_relative_improvement(dummy_prediction, per_class):
    score = compute_relative_improvement(dummy_prediction, per_class=per_class)
    assert not score.isnull().values.any()
