from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score

import eeg_augmentation_benchmark
from eeg_augmentation_benchmark.scoring import compute_relative_improvement
from plots.plot_utils import setup_style, get_tfs_names
from plots.plot_utils import infer_task


time_palette = sns.color_palette("Greens")
freq_palette = sns.color_palette("Blues")
spatial_palette = sns.color_palette("Oranges")

HUE_MAP = {
    'IdentityTransform': [0.2, 0.2, 0.2, 0.5],
    'No augmentation': [0.2, 0.2, 0.2, 0.5],
    'FTSurrogate': freq_palette[5],
    'FrequencyShift': freq_palette[4],
    'BandstopFilter': freq_palette[3],
    'GaussianNoise': time_palette[2],
    'SmoothTimeMask': time_palette[5],
    'TimeReverse': time_palette[4],
    'SignFlip': time_palette[3],
    'ChannelsDropout': spatial_palette[3],
    'ChannelsShuffle': spatial_palette[2],
    'ChannelsSymmetry': spatial_palette[1],
    'SensorsZRotation': spatial_palette[4],
}


def new_plot_learning_curve(df, fig_save_path="", lw=0.8):
    # Compute scores
    test_df = df[df['set'] == 'test'].copy()
    # test_scores = compute_score(test_df, metric=balanced_accuracy_score)
    test_scores = compute_relative_improvement(
        test_df, metric=balanced_accuracy_score, percent=True)

    # rename the x axis to show powers of 2
    test_scores['power'] = test_scores.apply(
        lambda x: '$2^{' + str(int(np.log2(x.proportion))) + '}$',
        axis=1
    )
    test_scores.augmentation = test_scores.augmentation.apply(
        lambda x: x.replace("()", "")
    )

    test_scores.augmentation = test_scores.augmentation.apply(
        lambda x: x.replace("IdentityTransform", "No augmentation")
    )

    # Plot
    _, ax = plt.subplots()
    sns.pointplot(
        x='power',
        y='score',
        data=test_scores,
        hue='augmentation',
        markers=['o', 's'] * 3,
        linestyles=['-', '--'] * 3,
        dodge=True,
        palette=HUE_MAP,
        ax=ax,
        lw=lw,
    )
    plt.hlines(0, *ax.get_xlim(), linestyle="dashdot", color="k", lw=lw*1.5)
    ax.set_xlabel('Training set proportion')
    ax.set_ylabel('Test accuracy relative\n improvement (%)')
    _, labels = ax.get_legend_handles_labels()
    handles = plt.gca().get_lines()
    ax.legend(
        labels=labels,
        handles=handles[::len(handles) // 6],
        ncol=3,
        loc="upper right"
    )
    plt.tight_layout()

    # Save
    if len(fig_save_path) > 0:
        plt.savefig(fig_save_path)
        plt.show()


BEST_TRANSFORMS = {
    "physionet": [
        "IdentityTransform()",
        "TimeReverse()",
        "SignFlip()",
        "FTSurrogate()",
        "FrequencyShift()",
        "ChannelsDropout()",
        "ChannelsShuffle()",
    ],
    "BCI": [
        "IdentityTransform()",
        "TimeReverse()",
        "SmoothTimeMask()",
        "FTSurrogate()",
        "BandstopFilter()",
        "ChannelsDropout()",
        "SensorsZRotation()",
    ]
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Learning curve plotter",
        "Plots final summary learning curve results for all categories"
    )

    parser.add_argument(
        "fname",
        type=str,
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="",
    )

    parser.add_argument(
        "--png",
        action="store_true",
    )

    parser.add_argument(
        "--col",
        action="store_true",
    )

    args = parser.parse_args()

    # Figure parameters
    setup_style(grid=True, column_fig=args.col)

    # Load data
    df = pd.read_pickle(args.fname)
    for tf_cat in ["time", "frequency", "sensors", "rotations"]:
        aug_names = get_tfs_names(df, tf_cat)
        for aug in aug_names:
            df.loc[df["augmentation"] == aug, "type"] = tf_cat

        # Infer task type
        task_properties = infer_task(df)

        ordered_df = pd.concat(
            [
                df.query("augmentation == @aug")
                for aug in BEST_TRANSFORMS[task_properties["dataset"]]
            ],
            ignore_index=True,
        )

        # Create saving path
        ext = "png" if args.png else "pdf"
        ds = task_properties["dataset"]
        fig_dir = Path(eeg_augmentation_benchmark.__file__).parent / (
            f'../outputs/{ds}/figures/'
        )
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Plot and save learning curve
        fig_path_lr = fig_dir / f'LR-{ds}-all-rel{args.suffix}.{ext}'
        new_plot_learning_curve(
            ordered_df,
            fig_save_path=str(fig_path_lr),
        )
