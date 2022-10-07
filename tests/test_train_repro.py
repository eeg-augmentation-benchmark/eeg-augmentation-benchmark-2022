import copy

import numpy as np
import pytest
from braindecode.augmentation import TimeReverse
from sklearn.model_selection import train_test_split

from BAE.train import get_clf
from BAE.train import get_dataset
from BAE.training_utils import cross_val_aug
from BAE.training_utils import fit_and_predict
from BAE.training_utils import fit_and_predict_one_subject
from BAE.training_utils import parallel_train_subjects
from BAE.utils import downsample


def test_fit_and_predict(random_state, cachedir):
    """Lower level test.
    """
    dataset = get_dataset(
        name='BCI',
        n_subjects=4,
        n_jobs=2,
        cachedir=cachedir)
    clf = get_clf(name='BCI')
    train_indices, test_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2
    )
    prediction_1 = fit_and_predict(
        clf=copy.deepcopy(clf),
        dataset=dataset,
        train_indices=train_indices,
        test_indices=test_indices,
        epochs=2,
        proportion=0.1,
        random_state=random_state,
        subjects_mask=np.arange(len(dataset)),  # fake subjects
        unbalanced=True
    )

    prediction_2 = fit_and_predict(
        clf=copy.deepcopy(clf),
        dataset=dataset,
        train_indices=train_indices,
        test_indices=test_indices,
        epochs=2,
        proportion=0.1,
        random_state=random_state,
        subjects_mask=np.arange(len(dataset)),  # fake subjects
        unbalanced=True
    )
    for i in range(len(prediction_1)):
        assert np.array_equal(
            prediction_1['y_pred'][i],
            prediction_2['y_pred'][i]
        )


@pytest.mark.parametrize("name", [
    'BCI', 'SleepPhysionet',
])
def test_CV(name, random_state, cachedir):
    """Parallelization test.
    """
    dataset = get_dataset(
        name=name,
        n_subjects=4,
        n_jobs=2,
        recording_ids=[1],
        cachedir=cachedir
    )
    clf1 = get_clf(name=name, random_state=random_state)
    if clf1.device.type == 'cuda':
        n_jobs = 4
    else:
        n_jobs = 2

    windows1, subjects_mask = downsample(dataset, random_state=random_state)

    prediction_1 = cross_val_aug(
        clf=copy.deepcopy(clf1),
        dataset=windows1,
        subjects_mask=subjects_mask,
        K=2,
        epochs=2,
        augmentations=[
            TimeReverse(probability=0.5, random_state=random_state)
        ],
        n_jobs=n_jobs,
        proportion=0.05,
        random_state=random_state,
        valid_size=0.5,
        unbalanced=True,
    )

    clf2 = get_clf(name=name, random_state=random_state)

    windows2, subjects_mask = downsample(dataset, random_state=random_state)
    prediction_2 = cross_val_aug(
        clf=copy.deepcopy(clf2),
        dataset=windows2,
        subjects_mask=subjects_mask,
        K=2,
        epochs=2,
        augmentations=[
            TimeReverse(probability=0.5, random_state=random_state)
        ],
        n_jobs=n_jobs,
        proportion=0.05,
        random_state=random_state,
        valid_size=0.5,
        unbalanced=True,
    )

    for i in range(len(prediction_2)):
        assert np.array_equal(
            prediction_1['y_pred'][i],
            prediction_2['y_pred'][i]
        )


def test_train_one_subject(random_state, cachedir):

    dataset = get_dataset(
        name='BCI',
        n_subjects=4,
        n_jobs=2,
        recording_ids=[1],
        cachedir=cachedir
    )
    subject_split = dataset.split('subject')
    subject_id = list(subject_split.keys())[0]
    clf = get_clf('BCI')

    prediction_1 = fit_and_predict_one_subject(
        dataset,
        copy.deepcopy(clf),
        subject_id,
        epochs=2,
        random_state_subject=random_state
    )
    prediction_2 = fit_and_predict_one_subject(
        dataset,
        copy.deepcopy(clf),
        subject_id,
        epochs=2,
        random_state_subject=random_state
    )
    for i in range(len(prediction_2)):
        assert np.array_equal(
            prediction_1['y_pred'][i],
            prediction_2['y_pred'][i]
        )


def test_parallel_BCI(random_state, cachedir):

    dataset = get_dataset(
        name='BCI',
        n_subjects=4,
        n_jobs=2,
        recording_ids=[1],
        cachedir=cachedir
    )
    clf = get_clf('BCI')
    prediction_1 = parallel_train_subjects(
        dataset,
        copy.deepcopy(clf),
        n_jobs=2,
        epochs=2,
        proportion=0.05,
        random_state=random_state
    )
    prediction_2 = parallel_train_subjects(
        dataset,
        copy.deepcopy(clf),
        n_jobs=2,
        epochs=2,
        proportion=0.05,
        random_state=random_state
    )
    for i in range(len(prediction_2)):
        assert np.array_equal(
            prediction_1['y_pred'][i],
            prediction_2['y_pred'][i]
        )
