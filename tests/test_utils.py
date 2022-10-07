
import numpy as np
from braindecode.augmentation import AugmentedDataLoader
from braindecode.augmentation import SignFlip
from braindecode.augmentation import TimeReverse

from BAE.utils import ClassWiseAugmentation


def test_classwise_aug(mock_dataset, random_state):
    rng = np.random.RandomState(random_state)
    aug_dict = {
        0: SignFlip(
            probability=1,
            random_state=rng),
        1: SignFlip(
            probability=1,
            random_state=rng),
        2: SignFlip(
            probability=1,
            random_state=rng),
        3: SignFlip(
            probability=1,
            random_state=rng),
        4: TimeReverse(
            probability=1,
            random_state=rng)
    }
    augmentation = ClassWiseAugmentation(aug_dict)

    dataloader_cw = AugmentedDataLoader(
        mock_dataset, transforms=[augmentation], batch_size=2)
    dataloader_SignFlip = AugmentedDataLoader(
        mock_dataset, transforms=[aug_dict[0]], batch_size=2)
    dataloader_TimeReverse = AugmentedDataLoader(
        mock_dataset, transforms=[aug_dict[4]], batch_size=2)

    for batch_TR, batch_SF, batch_CW in zip(
            dataloader_TimeReverse, dataloader_SignFlip, dataloader_cw):
        y = batch_CW[1]
        for i, label in enumerate(y):
            # if label==4 TimeReverse is applied.
            if int(label) == 4:
                X_1, X_2 = batch_TR[i][0], batch_CW[i][0]
                assert X_1.shape == X_2.shape
                assert np.array_equal(X_1, X_2)
            # else SignFlip is applied
            else:
                X_1, X_2 = batch_SF[i][0], batch_CW[i][0]
                assert X_1.shape == X_2.shape
                assert np.array_equal(X_1, X_2)
