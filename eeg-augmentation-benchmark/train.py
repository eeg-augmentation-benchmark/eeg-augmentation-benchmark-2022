import time
from pathlib import Path

import numpy as np
from braindecode.augmentation import ChannelsDropout
from braindecode.augmentation import ChannelsShuffle
from braindecode.augmentation import ChannelsSymmetry
from braindecode.augmentation import FrequencyShift
from braindecode.augmentation import FTSurrogate
from braindecode.augmentation import GaussianNoise
from braindecode.augmentation import IdentityTransform
from braindecode.augmentation import SignFlip
from braindecode.augmentation import SmoothTimeMask
from braindecode.augmentation import TimeReverse
from braindecode.augmentation.transforms import BandstopFilter
from braindecode.augmentation.transforms import SensorsXRotation
from braindecode.augmentation.transforms import SensorsYRotation
from braindecode.augmentation.transforms import SensorsZRotation
from braindecode.datasets import MOABBDataset
from braindecode.datasets import SleepPhysionet
from braindecode.datautil.serialization import load_concat_dataset
from braindecode.models import ShallowFBCSPNet
from braindecode.models import SleepStagerChambon2018
from braindecode.models import to_dense_prediction_model
from braindecode.preprocessing import Preprocessor
from braindecode.preprocessing import exponential_moving_standardize
from braindecode.preprocessing import preprocess
from sklearn.preprocessing import scale

from BAE.models import CLF_PARAMS_BCI
from BAE.models import CLF_PARAMS_BCI_CROP
from BAE.models import CLF_PARAMS_PHYSIONET
from BAE.models import MODEL_PARAMS_BCI
from BAE.models import MODEL_PARAMS_BCI_CROP
from BAE.models import MODEL_PARAMS_PHYSIONET
from BAE.models import get_EEGClassifier
from BAE.utils import EEGDataset


BEST_PARAMS = {
    "FrequencyShift": {
        "BCI": 2.7,
        "SleepPhysionet": 0.3,
    },
    "FTSurrogate": {
        "BCI": 0.9,
        "SleepPhysionet": 0.8,
    },
    "BandstopFilter": {
        "BCI": 0.4,
        "SleepPhysionet": 1.2,
    },
    "GaussianNoise": {
        "BCI": 0.16,
        "SleepPhysionet": 0.12,
    },
    "SmoothTimeMask": {
        "BCI": 400,
        "SleepPhysionet": 200,
    },
    "ChannelsDropout": {
        "BCI": 1.,
        "SleepPhysionet": 0.4,
    },
    "ChannelsShuffle": {
        "BCI": 0.1,
        "SleepPhysionet": 0.8,
    },
    "SensorsXRotation": {
        "BCI": 3,
        "SleepPhysionet": 25,
    },
    "SensorsYRotation": {
        "BCI": 12,
        "SleepPhysionet": 9,
    },
    "SensorsZRotation": {
        "BCI": 3,
        "SleepPhysionet": 30,
    },
}


def get_dataset(name, n_subjects, n_jobs=1, recording_ids=None, preload=True,
                cachedir=None, exp_mov_std=True, fmin=None, fmax=None,
                tmin=None, tmax=None):
    """
    Load a dataset. If the user wants to add a new dataset with a specific
    preprocessing pipeline, it can be done in this function.

    Parameters:
    -----------
    name: str
        Name of the dataset that will be used as an argparse argument in
        make_learning_curve.py.
    n_subjects: int
        Number of subjects to extract from the dataset.
    n_jobs:
        Number of workers for the parallelisation of the windowing.
    cachedir: str | None
        XXX

    Returns:
    --------
    windows: BaseConcatDataset:
        Preprocessed windows, ready for the training !

    Note
    ----
    The Sleep EDF contains 78 subjects but their ideas are not evenly spaced
    in [0, 77] (it vould be too easy). Instead, indices range from 0 to 82 with
    [39, 68, 69, 78, 79] missing. This dataset implementation makes up for this
    peculiarity. Though this fact must be acknowledged if you are looking for
    a specific subject ID.
    """
    # Dataset:
    dataset_dir_name = f"{name}_{n_subjects}"
    load_from_cache = cachedir is not None and (
        (Path(cachedir) / dataset_dir_name).exists()
    )
    if load_from_cache:
        windows = load_concat_dataset(Path(cachedir) / dataset_dir_name,
                                      preload=preload)
    elif name == "SleepPhysionet":
        SUBJECT_IDS = np.delete(np.arange(83), [39, 68, 69, 78, 79])

        dataset = EEGDataset(SleepPhysionet(
            subject_ids=SUBJECT_IDS[:n_subjects],
            recording_ids=recording_ids,
            preload=False,
            load_eeg_only=True
        ))
        # Preprocessing
        if fmax is None:
            fmax = 30
        preprocessors = [
            Preprocessor('pick', picks=['Fpz-Cz', 'Pz-Oz']),
            Preprocessor('load_data'),
            Preprocessor(lambda x: x * 1e6),
            Preprocessor('filter', l_freq=None, h_freq=fmax),
        ]
        t0 = time.time()
        preprocess(dataset, preprocessors)
        print(f"Raw preproc time: {time.time() - t0}")

        windows = dataset.get_windows(
            mapping={  # We merge stages 3 and 4 following AASM standards.
                'Sleep stage W': 0,
                'Sleep stage 1': 1,
                'Sleep stage 2': 2,
                'Sleep stage 3': 3,
                'Sleep stage 4': 3,
                'Sleep stage R': 4},
            window_size_samples=3000,
            window_stride_samples=3000,
            preload=preload,
            n_jobs=n_jobs,
            picks=['Fpz-Cz', 'Pz-Oz']
        )
        t0 = time.time()
        preprocess(windows, [Preprocessor(fn=scale, channel_wise=True)])
        print(f"Windows preproc time: {time.time() - t0}")

    elif name == "BCI":
        if n_subjects > 9:
            n_subjects = 9
        dataset = EEGDataset(
            MOABBDataset(
                dataset_name="BNCI2014001",
                # Subjects are indexed from 1 to 9
                subject_ids=list(np.arange(n_subjects) + 1)))
        factor_new = 1e-3
        init_block_size = 1000
        if fmin is None:
            fmin = 4
        if fmax is None:
            fmax = 38
        preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),
            # Keep EEG sensors
            Preprocessor(lambda x: x * 1e6),            # Convert from V to uV
            Preprocessor('filter', l_freq=fmin, h_freq=fmax),  # Bandpass
        ]
        if exp_mov_std:
            preprocessors.append(
                Preprocessor(
                    exponential_moving_standardize,
                    factor_new=factor_new, init_block_size=init_block_size
                )
            )

        preprocess(dataset, preprocessors)

        if tmin is None:
            trial_start_offset_seconds = -0.5
        else:
            trial_start_offset_seconds = tmin
        # Extract sampling frequency, check that they are same in all datasets
        sfreq = dataset.datasets[0].raw.info['sfreq']
        assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
        tmax_samples = int(tmax * sfreq) if tmax is not None else 0
        windows = dataset.get_windows(
            mapping={'feet': 0,
                     'left_hand': 1,
                     'right_hand': 2,
                     'tongue': 3},
            preload=preload,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=tmax_samples,
            n_jobs=n_jobs)

    elif name == "BCI_CROP":
        if n_subjects > 9:
            n_subjects = 9
        dataset = EEGDataset(
            MOABBDataset(
                dataset_name="BNCI2014001",
                # Subjects are indexed from 1 to 9
                subject_ids=list(np.arange(n_subjects) + 1)))
        factor_new = 1e-3
        init_block_size = 1000
        preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),
            # Keep EEG sensors
            Preprocessor(lambda x: x * 1e6),            # Convert from V to uV
            Preprocessor('filter', l_freq=4, h_freq=38),
            # Bandpass filter
            Preprocessor(exponential_moving_standardize,
                         factor_new=factor_new, init_block_size=init_block_size
                         )
        ]
        preprocess(dataset, preprocessors)

        trial_start_offset_seconds = -0.5
        # Extract sampling frequency, check that they are same in all datasets
        sfreq = dataset.datasets[0].raw.info['sfreq']
        assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
        input_window_samples = 1000
        # depends on the model used for classification
        n_preds_per_input = 467

        windows = dataset.get_windows(
            mapping={'feet': 0,
                     'left_hand': 1,
                     'right_hand': 2,
                     'tongue': 3},
            preload=preload,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            window_size_samples=input_window_samples,
            window_stride_samples=n_preds_per_input,
            drop_last_window=False,
            n_jobs=n_jobs,
        )
    if not load_from_cache:
        if cachedir is None:
            cachedir = '__cache__'
        output_dir = Path(cachedir) / dataset_dir_name
        output_dir.mkdir(exist_ok=True, parents=True)
        windows.save(Path(cachedir) / dataset_dir_name, overwrite=True)

    return windows


def get_augmentations_list(
    names,
    probability,
    windows,
    dataset,
    random_state=None
):
    """
    Parameters
    ----------
    names: list of str
        List of augmentation names.
    probability: float
        Probability to apply augmentations when data is loaded.
    windows: WindowsDataset
        Windows dataset used for the training. Allows to determine several
        parameters such as the sampling rate.
    dataset : str,
        Either SleepPhysionet or BCI.
    random_state: int | None | RandomGenerator, optional
        Seed or random number generator.

    Returns
    -------
    aug_list : list
        The list of instances of augmentations.
    """
    if dataset == "BCI_CROP":
        dataset = "BCI"

    # put int
    rng = np.random.RandomState(random_state)
    aug_list = []
    for name in names:
        if name == 'GaussianNoise':
            aug_list.append(GaussianNoise(
                probability=probability,
                std=BEST_PARAMS[name][dataset],
                random_state=rng))
        elif name == 'FTSurrogate':
            if dataset == "SleepPhysionet":
                channel_indep = True
            else:
                channel_indep = False
            aug_list.append(FTSurrogate(
                probability=probability,
                random_state=rng,
                phase_noise_magnitude=BEST_PARAMS[name][dataset],
                channel_indep=channel_indep,))
        elif name == 'SignFlip':
            aug_list.append(SignFlip(
                probability=probability,
                random_state=rng))
        elif name == 'SmoothTimeMask':
            aug_list.append(SmoothTimeMask(
                probability=probability,
                mask_len_samples=BEST_PARAMS[name][dataset],
                random_state=rng))
        elif name == "TimeReverse":
            aug_list.append(TimeReverse(
                probability=probability,
                random_state=rng))
        elif name == "FrequencyShift":
            aug_list.append(FrequencyShift(
                probability=probability,
                sfreq=windows.datasets[0].windows.info['sfreq'],
                max_delta_freq=BEST_PARAMS[name][dataset],
                random_state=rng))
        elif name == "IdentityTransform":
            aug_list.append(IdentityTransform())
        elif name == "ChannelsDropout":
            aug_list.append(ChannelsDropout(
                probability=probability,
                p_drop=BEST_PARAMS[name][dataset],
                random_state=rng))
        elif name == "ChannelsShuffle":
            aug_list.append(ChannelsShuffle(
                probability=probability,
                p_shuffle=BEST_PARAMS[name][dataset],
                random_state=rng))
        elif name == "ChannelsSymmetry":
            aug_list.append(ChannelsSymmetry(
                probability=probability,
                ordered_ch_names=windows.datasets[0].windows.ch_names,
                random_state=rng))
        elif name == "BandstopFilter":
            aug_list.append(BandstopFilter(
                probability=probability,
                sfreq=windows.datasets[0].windows.info['sfreq'],
                bandwidth=BEST_PARAMS[name][dataset],
                max_freq=30,
                random_state=rng))
        elif name == "SensorsXRotation":
            aug_list.append(SensorsXRotation(
                probability=probability,
                ordered_ch_names=windows.datasets[0].windows.ch_names,
                max_degrees=BEST_PARAMS[name][dataset],
                random_state=rng))
        elif name == "SensorsYRotation":
            aug_list.append(SensorsYRotation(
                probability=probability,
                ordered_ch_names=windows.datasets[0].windows.ch_names,
                max_degrees=BEST_PARAMS[name][dataset],
                random_state=rng))
        elif name == "SensorsZRotation":
            aug_list.append(SensorsZRotation(
                probability=probability,
                ordered_ch_names=windows.datasets[0].windows.ch_names,
                max_degrees=BEST_PARAMS[name][dataset],
                random_state=rng))
    return aug_list


def get_clf(name, random_state=None):
    if name == 'SleepPhysionet':
        return get_EEGClassifier(
            model=SleepStagerChambon2018,
            model_params=MODEL_PARAMS_PHYSIONET,
            clf_params=CLF_PARAMS_PHYSIONET,
            random_state=random_state,
        )
    elif name == 'BCI':
        return get_EEGClassifier(
            model=ShallowFBCSPNet,
            model_params=MODEL_PARAMS_BCI,
            clf_params=CLF_PARAMS_BCI,
            random_state=random_state,
        )
    elif name == 'BCI_CROP':
        clf = get_EEGClassifier(
            model=ShallowFBCSPNet,
            model_params=MODEL_PARAMS_BCI_CROP,
            clf_params=CLF_PARAMS_BCI_CROP,
            random_state=random_state,
        )
        to_dense_prediction_model(clf.module)
        return clf
    raise Exception('Classifier not found')
