import copy

import numpy as np
from BAE.train import get_clf
from BAE.train import get_dataset
from BAE.training_utils import parallel_train_subjects_crop
from braindecode.augmentation import TimeReverse


def test_parallel_train_subjects_crop(cachedir, random_state):
    dataset = get_dataset(
        name="BCI_CROP",
        n_subjects=2,
        n_jobs=2,
        recording_ids=[1],
        cachedir=cachedir
    )

    clf1 = get_clf(name="BCI_CROP", random_state=random_state)
    clf2 = copy.deepcopy(clf1)
    if clf1.device.type == 'cuda':
        n_jobs = 4
    else:
        n_jobs = 2

    prediction_1 = parallel_train_subjects_crop(
        dataset=dataset,
        clf=copy.deepcopy(clf1),
        n_jobs=n_jobs,
        augmentations=[
            TimeReverse(probability=0.5, random_state=random_state)
        ],
        random_state=random_state,
    )

    prediction_2 = parallel_train_subjects_crop(
        dataset=dataset,
        clf=clf2,
        n_jobs=n_jobs,
        augmentations=[
            TimeReverse(probability=0.5, random_state=random_state)
        ],
        random_state=random_state,
    )

    for i in range(len(prediction_2)):
        assert np.array_equal(
            prediction_1['y_pred'][i],
            prediction_2['y_pred'][i]
        )
