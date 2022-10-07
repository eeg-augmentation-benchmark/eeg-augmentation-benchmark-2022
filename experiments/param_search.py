import time
from argparse import ArgumentDefaultsHelpFormatter
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import braindecode.augmentation as Augmentations
import numpy as np
import pandas as pd
from BAE.param_search_utils import DEFAULT_AUG_PARAMS
from BAE.param_search_utils import get_augmentation
from BAE.train import get_clf
from BAE.train import get_dataset
from BAE.training_utils import cross_val_aug
from BAE.training_utils import parallel_train_subjects
from BAE.training_utils import parallel_train_subjects_crop
from BAE.utils import downsample
from BAE.utils import find_device
from BAE.utils import get_subjects

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
                    default=2,
                    help="Number of folds used for"
                    "cross-validation training")
parser.add_argument('-n', '--subjects',
                    type=int,
                    default=5,
                    help="Number of subjects used to create"
                    "the whole dataset")
parser.add_argument('--proportion',
                    type=int,
                    # gives ~200 windows when using 40 subjects (reduce std)
                    default=8,  # for 1 / 2**8
                    help="Dataset proportion used for the training.")
parser.add_argument('-t', '--transformations',
                    type=str,
                    default=[
                        "FTSurrogate",
                        "GaussianNoise",
                        "SmoothTimeMask",
                        "FrequencyShift",
                        "IdentityTransform",
                        "BandstopFilter",
                        "ChannelsDropout",
                        "ChannelsShuffle",
                        'SensorsXRotation',
                        'SensorsYRotation',
                        'SensorsZRotation'
                    ],
                    nargs='+',
                    help="List of transformations to plot "
                    "can be chosen among: [FTSurrogate, GaussianNoise, "
                    "SignFlip, SmoothTimeMask, TimeReverse, FrequencyShift, "
                    "IdentityTransform, SensorsXRotation, SensorsYRotation, "
                    "SensorsZRotation]")
parser.add_argument('-m', '--magnitudes',
                    type=float,
                    default=np.linspace(0.1, 1., 10),
                    nargs='+',
                    help="dataset fractions used for the learning curve")
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
parser.add_argument('--cachedir',
                    type=str,
                    default=None,
                    help='Path where to store preprocessed data.')
parser.add_argument('--device',
                    type=str,
                    default=None,
                    help='Device to use, default None will use CPU or cuda:1')
parser.add_argument('--downsampling',
                    action='store_true',
                    default=False,
                    help='Whether to downsample the training set so that all'
                    'classes are balanced.')


def main():
    args = parser.parse_args()
    print(args)
    print(f"\n\n{1 / 2**args.proportion}")

    # List of augmentations with diff magnitudes
    def get_aug_mag(magnitude):
        aug_list = []
        for transfo in args.transformations:
            aug = getattr(Augmentations, transfo)
            aug_params = DEFAULT_AUG_PARAMS[args.dataset][transfo]
            aug_list.append(
                get_augmentation(
                    aug,
                    magnitude=magnitude,
                    random_state=args.random_state,
                    **aug_params
                )
            )
        return aug_list

    # Dataset
    windows = get_dataset(
        name=args.dataset,
        n_subjects=args.subjects,
        n_jobs=args.n_jobs,
        cachedir=args.cachedir
    )

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

    # Model
    device, _ = find_device(args.device)
    clf = get_clf(name=args.dataset, random_state=args.random_state)
    clf.set_params(device=device)
    print('\nDevice: {}\n'.format(clf.device))

    t_start = perf_counter()
    output_df = []
    for magnitude in args.magnitudes:
        print('\nmag:{}'.format(magnitude))
        aug_list = get_aug_mag(magnitude)
        this_output_df = train_func(
            clf=clf,
            dataset=windows,
            subjects_mask=subjects_mask,
            K=args.folds,
            epochs=args.epochs,
            augmentations=aug_list,
            n_jobs=args.n_jobs,
            proportion=1 / 2**args.proportion,
            random_state=args.random_state
        )
        this_output_df['magnitude'] = magnitude
        output_df.append(this_output_df)
        # checkpoint
        if args.output_dir:
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            current_time = time.strftime("%b%d-%Hh%M")
            checkpoint_df = pd.concat(output_df, axis=0)
            checkpoint_df.to_pickle(
                output / str('checkpoint-mag{}-'.format(magnitude)
                             + current_time + '.pkl'))
    output_df = pd.concat(output_df, axis=0)
    t_stop = perf_counter()

    print('\nExperiment duration: {}s\n'.format(t_stop - t_start))
    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        current_time = time.strftime("%b%d-%Hh%M")
        sname = args.save_name if args.save_name else str(
            '-'.join(args.transformations) + '-' + current_time
        )
        output_df.to_pickle(output / sname + '.pkl')


if __name__ == "__main__":
    main()
