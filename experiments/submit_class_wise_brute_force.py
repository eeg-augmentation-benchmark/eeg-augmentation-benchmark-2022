from argparse import ArgumentDefaultsHelpFormatter
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
from time import strftime
import os

from BAE.class_wise_aug import prepare_aug_list
from BAE.train import get_clf
from BAE.train import get_dataset
from BAE.training_utils import cross_val_aug
from BAE.utils import downsample
from BAE.utils import get_subjects
from cluster_config import CLUSTER_CONFIGS


def run_one(args, sub_list_augs, i):

    epochs = args.epochs
    n_subjects = args.subjects
    folds = args.folds

    if args.debug:
        epochs = 3
        n_subjects = 10
        folds = 2

    windows = get_dataset(
        name=args.dataset,
        n_subjects=n_subjects,
        n_jobs=int(args.n_jobs / 2),
        preload=False,
    )

    # Downsampling
    if args.downsampling:
        windows, subjects_mask = downsample(
            windows, random_state=args.random_state
        )
    else:
        subjects_mask = get_subjects(windows)

    # Model
    clf = get_clf(name=args.dataset)

    t_start = perf_counter()

    output_df = cross_val_aug(
        clf=clf,
        dataset=windows,
        subjects_mask=subjects_mask,
        K=folds,
        epochs=epochs,
        augmentations=sub_list_augs,
        n_jobs=args.n_jobs,
        proportion=args.proportions,
        random_state=args.random_state,
    )
    t_stop = perf_counter()
    print('\nExperiment duration: {}s\n'.format(t_stop - t_start))

    if args.output:
        output = Path(args.output)
        os.makedirs(output, exist_ok=True)
        current_time = strftime("%b%d-%Hh%M")
        save_file_name = 'class_wise_brut_force_results_part{}-{}.pkl'.format(
            i, current_time,
        )
        output_df.to_pickle(output / save_file_name)


if __name__ == '__main__':
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
                        # np.logspace(3, 12, 10, base=1 / 2)
                        # slightly under 2^-9 --> 2^6 windows
                        default=1 / 2**7,
                        help="Dataset proportion used for the training.")
    parser.add_argument('-j', '--n_jobs',
                        type=int,
                        default=1,
                        help="Number of processes for parallelization.")
    parser.add_argument('-r', '--random_state',
                        type=int,
                        default=19,
                        help='Set random state for reproductibility.')
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help='Path to the output folder')
    parser.add_argument('-d', '--dataset',
                        type=str,
                        default='SleepPhysionet',
                        help='Dataset to use. Can be either SleepPhysionet or'
                        'BCI')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use, default None will use CPU or cuda:1')
    parser.add_argument(
        '--downsampling',
        action='store_true',
        default=False,
        help='Whether to downsample the training set so that all'
        'classes are balanced.')
    parser.add_argument(
        '--n_gpus',
        default=1,
        type=int,
        help='Number of GPUs used for parallelization.')
    parser.add_argument(
        '--timeout',
        default=15,
        type=int,
        help='Computation timeout.')
    parser.add_argument(
        '--mem',
        default=100_000,
        type=int,
        help='Computation CPU memory.')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode')

    args = parser.parse_args()

    n_gpus = args.n_gpus
    aug_lists = prepare_aug_list(n_gpus=n_gpus)
    if args.debug:
        aug_lists = [[aug_lists[0][1]]]

    get_executor = CLUSTER_CONFIGS['jean-zay']
    job_name = 'class_wise_brute_force'
    executor = get_executor(
        job_name,
        timeout_hour=args.timeout,
        mem=args.mem,
    )
    print('submitting jobs...', end='', flush=True)
    with executor.batch():
        for i, sub_aug_list in enumerate(aug_lists):
            executor.submit(
                run_one,
                args,
                sub_aug_list,
                i
            )
    print('Done')
