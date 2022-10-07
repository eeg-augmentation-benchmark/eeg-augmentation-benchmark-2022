import braindecode.augmentation as Augmentations
import numpy as np
import pytest
from braindecode.augmentation import AugmentedDataLoader
from braindecode.augmentation import IdentityTransform

from eeg_augmentation_benchmark.param_search_utils import BCI_CHANNELS
from eeg_augmentation_benchmark.param_search_utils import DEFAULT_AUG_PARAMS
from eeg_augmentation_benchmark.param_search_utils import get_augmentation
from tests.conftest import MockDataset


@pytest.mark.parametrize("aug_name,params", [
    ('ChannelsDropout', {
        "probability": 0.5,
        "p_drop": 0.5
    }),
    ('GaussianNoise', {
        "probability": 0.5,
        "std": 0.1,
    }),
    ('FTSurrogate', {
        "probability": 0.5,
        "phase_noise_magnitude": 0.5
    }),
    ('SmoothTimeMask', {
        "probability": 0.5,
        "mask_len_samples": 250
    }),
    ('BandstopFilter', {
        "probability": 0.5,
        "bandwidth": 1.,
        "sfreq": 250,
        'max_freq': 30,
    }),
    ('ChannelsShuffle', {
        "probability": 0.5,
        "p_shuffle": 0.5
    }),
    ('FrequencyShift', {
        "probability": 0.5,
        "sfreq": 250,
        "max_delta_freq": 1.5
    }),
])
def test_get_augmentation(aug_name, params, random_state, mock_dataset):
    """Check that the output from `get_augmentation` is produces the same
    augmented samples that the expected transformation.
    """
    aug_class = getattr(Augmentations, aug_name)
    aug_expected = aug_class(random_state=random_state, **params)

    aug_params = DEFAULT_AUG_PARAMS["BCI"][aug_name]
    aug_instance = getattr(Augmentations, aug_name)
    aug = get_augmentation(
        aug_instance,
        magnitude=0.5,
        random_state=random_state,
        **aug_params
    )

    dataloader = AugmentedDataLoader(mock_dataset, transforms=[aug])
    dataloader_exp = AugmentedDataLoader(
        mock_dataset, transforms=[aug_expected])
    for X_1, X_2 in zip(dataloader_exp, dataloader):
        X_1, X_2 = X_1[0].squeeze(), X_2[0].squeeze()
        assert np.array_equal(X_1, X_2)


def test_get_augmentation_ref(random_state, mock_dataset):
    """
    Check that get_augmentation also works for IdentityTransform
    """
    aug_expected = IdentityTransform()
    aug = get_augmentation(IdentityTransform)
    for X in mock_dataset:
        expected_X_T = aug_expected(X[0])
        X_T = aug(X[0])
        assert np.array_equal(expected_X_T, X_T)


@pytest.mark.parametrize("aug_name,params", [
    ('SensorsZRotation', {
        "probability": 0.5,
        'ordered_ch_names': BCI_CHANNELS,
        "max_degrees": 15.
    }),
    ('SensorsYRotation', {
        "probability": 0.5,
        'ordered_ch_names': BCI_CHANNELS,
        "max_degrees": 15.
    }),
    ('SensorsXRotation', {
        "probability": 0.5,
        'ordered_ch_names': np.array(BCI_CHANNELS),
        "max_degrees": 15.
    }),
])
def test_get_rotations(aug_name, params, random_state):
    """Check that the output from `get_augmentation` is produces the same
    augmented samples that the expected transformation.
    """
    mock_dataset = MockDataset(50, n_channels=22)
    aug_class = getattr(Augmentations, aug_name)
    aug_expected = aug_class(random_state=random_state, **params)

    aug_params = DEFAULT_AUG_PARAMS["BCI"][aug_name]
    aug_instance = getattr(Augmentations, aug_name)
    aug = get_augmentation(
        aug_instance,
        magnitude=0.5,
        random_state=random_state,
        **aug_params
    )
    dataloader = AugmentedDataLoader(mock_dataset, transforms=[aug])
    dataloader_exp = AugmentedDataLoader(
        mock_dataset, transforms=[aug_expected])
    for X_1, X_2 in zip(dataloader_exp, dataloader):
        X_1, X_2 = X_1[0].squeeze(), X_2[0].squeeze()
        assert np.array_equal(X_1, X_2)
