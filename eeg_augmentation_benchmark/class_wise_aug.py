# %%
from itertools import combinations
from itertools import product

import braindecode.augmentation as Augmentations
import numpy as np

from eeg_augmentation_benchmark.utils import ClassWiseAugmentation

BEST_AUG_PARAMS = {
    'GaussianNoise': {
        'std': 0.16,
    },
    'FrequencyShift': {
        'max_delta_freq': 0.6,
        'sfreq': 100
    },
    'FTSurrogate': {
        'phase_noise_magnitude': 1,
    },
    'SmoothTimeMask': {
        'mask_len_samples': 140,
    },
    'ChannelsDropout': {
        'p_drop': 0.2,
    },
    'ChannelsShuffle': {
        'p_shuffle': 0.6,
    },
    'IdentityTransform': {
    },
    'BandstopFilter': {
        'bandwidth': 0.6,
        'sfreq': 100
    },
    'ChannelsSymmetry': {
        'ordered_ch_names': ['Fz', 'Pz']
    }
}


def get_CW_aug(aug_list, probability=0.5, random_state=None):
    aug_dict = {}
    for i, aug_name in enumerate(aug_list):
        augmentation_cls = getattr(Augmentations, aug_name)
        if aug_name in list(BEST_AUG_PARAMS.keys()):
            aug_params = BEST_AUG_PARAMS[aug_name]
        else:
            aug_params = {}

        aug_dict[i] = augmentation_cls(
            probability,
            random_state=random_state,
            **aug_params
        )
    return ClassWiseAugmentation(aug_dict)


def get_aug_grid():
    list_aug_names = [
        'TimeReverse',
        'SignFlip',
        'FTSurrogate',
        'ChannelsShuffle',
        'ChannelsDropout',
        'GaussianNoise',
        'ChannelsSymmetry',
        'SmoothTimeMask',
        'BandstopFilter',
        'FrequencyShift',
        'IdentityTransform'
    ]
    pairs = combinations(list_aug_names, 2)
    grid = []
    for pair in pairs:
        aug_lists = product(pair, repeat=5)
        for aug_list in aug_lists:
            # remove class agnostic augmentations.
            if len(np.unique(aug_list)) == 1:
                continue
            grid.append(list(aug_list))

    return grid


def prepare_aug_list(n_gpus=1):
    grid_names = get_aug_grid()
    grid_augs = [get_CW_aug(aug_list) for aug_list in grid_names]
    return np.array_split(grid_augs, n_gpus)
