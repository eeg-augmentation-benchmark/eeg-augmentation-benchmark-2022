import time
from argparse import ArgumentDefaultsHelpFormatter
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from eeg_augmentation_benchmark.train import get_augmentations_list
from eeg_augmentation_benchmark.train import get_clf
from eeg_augmentation_benchmark.train import get_dataset
from eeg_augmentation_benchmark.training_utils import cross_val_aug
from eeg_augmentation_benchmark.training_utils import parallel_train_subjects
from eeg_augmentation_benchmark.training_utils import (
    parallel_train_subjects_crop
)
from eeg_augmentation_benchmark.utils import downsample
from eeg_augmentation_benchmark.utils import find_device
from eeg_augmentation_benchmark.utils import get_subjects
from skorch.callbacks import LRScheduler

parser = ArgumentParser(
    description='Cross validated training of EEG recordings using data'
    'augmentations.',
    formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('-e', '--epochs',
                    type=int,
                    default=2,
                    help="Number of epochs for each training.")
parser.add_argument('-k', '--folds',
                    type=int,
                    default=5,
                    help="Number of folds used for"
                    "cross-validation training")
parser.add_argument('-n', '--subjects',
                    type=int,
                    default=5,
                    help="Number of subjects used to create"
                    "the whole dataset")
parser.add_argument('--proportions',
                    type=float,
                    default=8,
                    help="Highest power of 1/2 used as a fraction of the"
                    "training set. The proportions correspond to"
                    "1 / 2**np.arange(args.proportions, -1, -1)")
parser.add_argument('-t', '--transformations',
                    type=str,
                    default=[
                        "IdentityTransform",
                        "FTSurrogate",
                        "GaussianNoise",
                        "SignFlip",
                        "SmoothTimeMask",
                        "TimeReverse",
                        "FrequencyShift",
                        "ChannelsDropout",
                        "ChannelsShuffle",
                        "BandstopFilter",
                        "ChannelsSymmetry",
                        "SensorsXRotation",
                        "SensorsYRotation",
                        "SensorsZRotation",
                    ],
                    nargs='+',
                    help="List of transformations to plot"
                    "can be chosen among: [FTSurrogate, GaussianNoise,"
                    "SignFlip, SmoothTimeMask, TimeReverse, FrequencyShift, "
                    "IdentityTransform]"
                    )
parser.add_argument('-j', '--n_jobs',
                    type=int,
                    default=1,
                    help="Number of processes for parallelization.")
parser.add_argument('-p', '--proba',
                    type=float,
                    default=0.5,
                    help="Probability to apply augmentations.")
parser.add_argument('-r', '--random_state',
                    type=int,
                    default=19,
                    help='Set random state for reproductibility.')
parser.add_argument('-o', '--output-dir',
                    type=str,
                    default=None,
                    help='Path to the output folder')
parser.add_argument('-s', '--save-name',
                    type=str,
                    default=None,
                    help='Fila name to use for saving the output')
parser.add_argument('-d', '--dataset',
                    type=str,
                    default='SleepPhysionet',
                    help='Dataset to use. Can be either SleepPhysionet or'
                    'BCI')
parser.add_argument('--device',
                    type=str,
                    default=None,
                    help='Device to use, default None will use CPU or cuda:1')
parser.add_argument('--cachedir',
                    type=str,
                    default=None,
                    help='Path where to store preprocessed data.')
parser.add_argument('--downsampling',
                    action='store_true',
                    default=False,
                    help='Whether to downsample the training set so that all'
                    'classes are balanced.')


def main():
    args = parser.parse_args()
    print('\nARGS: {}\n\n'.format(args))

    # Dataset:
    windows = get_dataset(
        name=args.dataset,
        n_subjects=args.subjects,
        n_jobs=args.n_jobs,
        cachedir=args.cachedir
    )

    # Augmentations
    aug_list = get_augmentations_list(
        args.transformations,
        probability=args.proba,
        windows=windows,
        dataset=args.dataset,
        random_state=args.random_state
    )

    # Model
    clf = get_clf(name=args.dataset)
    clf.device, _ = find_device(args.device)

    t_start = perf_counter()
    print('\n Auglist: {}'.format(aug_list))
    props_pred = []

    if args.dataset == 'SleepPhysionet':
        train_func = cross_val_aug
        # Downsampling
        if args.downsampling:
            windows, subjects_mask = downsample(
                windows, random_state=args.random_state
            )
        else:
            subjects_mask = get_subjects(windows)

    elif args.dataset == 'BCI':
        train_func = parallel_train_subjects
        subjects_mask = get_subjects(windows)

    elif args.dataset == 'BCI_CROP':
        train_func = parallel_train_subjects_crop
        subjects_mask = get_subjects(windows)
        clf.callbacks.append(
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=1600 - 1))
        )
        print(clf.callbacks)

    proportions = 1 / 2**np.arange(args.proportions, -1, -1)
    print("\nProportions: {}".format(proportions))
    for p in proportions:
        props_pred.append(train_func(
            clf=clf,
            dataset=windows,
            subjects_mask=subjects_mask,
            K=args.folds,
            epochs=args.epochs,
            augmentations=aug_list,
            n_jobs=args.n_jobs,
            proportion=p,
            random_state=args.random_state,
        ))
    t_stop = perf_counter()
    print('\nExperiment duration: {}s\n'.format(t_stop - t_start))

    df = pd.concat(props_pred)
    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        current_time = time.strftime("%b%d-%Hh%M")
        sname = args.save_name if args.save_name else str(
            '-'.join(args.transformations) + '-' + current_time
        )
        df.to_pickle(output / sname + '.pkl')


if __name__ == "__main__":
    main()
