
from braindecode.augmentation import AugmentedDataLoader

from BAE.train import get_augmentations_list
from BAE.train import get_dataset


def test_get_augmentations_list(random_state, mock_dataset, cachedir):
    dataset = 'SleepPhysionet'
    windows = get_dataset(
        name=dataset,
        n_subjects=4,
        n_jobs=1,
        recording_ids=[1],
        cachedir=cachedir
    )
    aug_list = get_augmentations_list(
        names=[
            "FTSurrogate",
            "GaussianNoise",
            "SignFlip",
            "SmoothTimeMask",
            "TimeReverse",
            "IdentityTransform",
            "ChannelsDropout",
            "ChannelsShuffle",
        ],
        probability=0.5,
        windows=windows,
        dataset=dataset,
        random_state=random_state
    )
    for t in aug_list:
        dataloader = AugmentedDataLoader(mock_dataset, transforms=[t])
        for X_t in dataloader:
            X_t = X_t[0]
            assert X_t[0].shape == mock_dataset[0][0].shape
