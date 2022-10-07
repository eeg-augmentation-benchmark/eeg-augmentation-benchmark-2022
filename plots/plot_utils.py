import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from braindecode.datasets import SleepPhysionet
from braindecode.datautil.serialization import load_concat_dataset
from braindecode.preprocessing import Preprocessor
from braindecode.preprocessing import preprocess
from mne.time_frequency.multitaper import psd_array_multitaper

from BAE.utils import EEGDataset

FONTSIZE = 8
# A4 width: 6.3 inches 2*1.25 margins --> 5.8 figures
W = 5.5


def setup_style(grid=False, column_fig=False):
    plt.style.use('seaborn-paper')

    if column_fig:
        plt.rcParams["figure.figsize"] = (W, W / 1.7)
    else:
        plt.rcParams["figure.figsize"] = (W, W / 3)
    plt.rcParams["axes.grid"] = grid
    fontsize = FONTSIZE * 2 if column_fig else FONTSIZE
    lw = 1.0 if column_fig else 0.5
    plt.rcParams.update({
        'font.size': fontsize,
        'legend.fontsize': 'x-small',
        'axes.labelsize': 'small',
        'xtick.labelsize': 'small',
        'ytick.labelsize': 'small',
        'axes.titlesize': 'medium',
        'lines.linewidth': lw,
    })


def plot_signal(window, ax, t_start=800, t_stop=1700, sfreq=100, **kwargs):
    time = np.arange(0, len(window)) / sfreq
    ax.plot(time[t_start:t_stop], window[t_start:t_stop], **kwargs)
    ax.set_ylabel('Voltage ($\\mu V$)', fontsize=FONTSIZE)
    ax.margins(x=0)
    return ax


def plot_psd(windows, ax, fmin=4, fmax=20, **kwargs):
    psds, freqs = psd_array_multitaper(
        windows, fmin=fmin, fmax=fmax, sfreq=100)
    psds = 10 * np.log10(psds)  # convert to dB
    psds_mean = psds.mean(0).mean(0)
    ax.plot(freqs, psds_mean, **kwargs)
    ax.set(xlabel='Frequency (Hz)', ylabel='PSD (dB)')
    return ax


def infer_task(df):
    classes = np.unique(df.y_true.iloc[0])
    task_properties = {
        "classes": classes,
    }
    if len(classes) == 5:  # sleep scoring
        task_properties["classes_names"] = ("W", "N1", "N2", 'N3', "REM")
        task_properties["dataset"] = "physionet"
        task_properties["class_type"] = '\nSleep stage'
        task_properties["proportion"] = 2 ** -7
    else:
        task_properties["classes_names"] = (
            "foot", "left\nhand", "right\nhand", "tongue")
        task_properties["dataset"] = "BCI"
        task_properties["class_type"] = 'Movement'
        task_properties["proportion"] = 1
    return task_properties


def make_fixed_palette():
    return {
        'IdentityTransform': [0.2, 0.2, 0.2, 0.5],
        'No augmentation': [0.2, 0.2, 0.2, 0.5],
        'FTSurrogate': 'tab:blue',
        'FrequencyShift': 'tab:orange',
        'BandstopFilter': 'tab:green',
        'GaussianNoise': 'tab:blue',
        'SmoothTimeMask': 'tab:orange',
        'TimeReverse': 'tab:green',
        'SignFlip': 'tab:purple',
        'ChannelsDropout': 'tab:blue',
        'ChannelsShuffle': 'tab:orange',
        'ChannelsSymmetry': 'tab:green',
        'SensorsXRotation': 'tab:blue',
        'SensorsYRotation': 'tab:orange',
        'SensorsZRotation': 'tab:green',
    }


FREQ_TRANSFORMS = [
    'FTSurrogate()',
    'FrequencyShift()',
    'BandstopFilter()',
]


TIME_TRANSFORMS = [
    'GaussianNoise()',
    'SmoothTimeMask()',
    'TimeReverse()',
    'SignFlip()',
]


SENSOR_TRANSFORMS = [
    'ChannelsDropout()',
    'ChannelsShuffle()',
    'ChannelsSymmetry()',
]


ROT_TRANSFORMS = [
    'SensorsXRotation()',
    'SensorsYRotation()',
    'SensorsZRotation()',
]


TRANSFORMS_BY_CATEGORY = {
    "time": ["IdentityTransform()"] + TIME_TRANSFORMS,
    "frequency": ["IdentityTransform()"] + FREQ_TRANSFORMS,
    "sensors": ["IdentityTransform()"] + SENSOR_TRANSFORMS,
    "rotations": ["IdentityTransform()"] + ROT_TRANSFORMS,
}


def get_tfs_names(df, tfs_category):
    all_tf_names = df["augmentation"].unique()
    category_tfs = TRANSFORMS_BY_CATEGORY[tfs_category]
    return set(all_tf_names).intersection(category_tfs)


def get_windows():
    # Load SleepPhysionet for plots, without scaling
    dataset_dir_name = "SleepPhysionet_78_plot"
    cachedir = "../tmp"
    load_from_cache = (Path(cachedir) / dataset_dir_name).exists()

    if load_from_cache:
        windows = load_concat_dataset(Path(cachedir) / dataset_dir_name,
                                      preload=True)
    else:
        SUBJECT_IDS = np.delete(np.arange(83), [39, 68, 69, 78, 79])

        dataset = EEGDataset(SleepPhysionet(
            subject_ids=SUBJECT_IDS[:78],
            recording_ids=[1],
            preload=False,
            load_eeg_only=True
        ))
        # Preprocessing
        preprocessors = [
            Preprocessor('pick', picks=['Fpz-Cz', 'Pz-Oz']),
            Preprocessor('load_data'),
            Preprocessor(lambda x: x * 1e6),
            Preprocessor('filter', l_freq=None, h_freq=30),
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
            preload=True,
            n_jobs=1,
            picks=['Fpz-Cz', 'Pz-Oz']
        )
        t0 = time.time()
        print(f"Windows preproc time: {time.time() - t0}")

    if not load_from_cache:
        output_dir = Path(cachedir) / dataset_dir_name
        output_dir.mkdir(exist_ok=True, parents=True)
        windows.save(Path(cachedir) / dataset_dir_name, overwrite=True)
    return windows
