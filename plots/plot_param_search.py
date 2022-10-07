# Imports
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import eeg_augmentation_benchmark
from eeg_augmentation_benchmark.scoring import compute_relative_improvement
from plots.plot_utils import setup_style, make_fixed_palette, get_tfs_names
from plots.plot_utils import infer_task


UNSCALED_MAG_RANGE = np.linspace(0, 0.9, 4)


MAG_RANGES = {
    'FTSurrogate': {
        'label': '$\\Delta \\varphi$',
        'ticks': np.linspace(0, 10, 5),
        # 'ticks_labels': ['0', '$\\frac{2}{3}\\pi$', '$\\frac{4}{3}\\pi$',
        #                 '2$\\pi$'],
        'ticks_labels': ['0', '$\\frac{\\pi}{2}$', '$\\pi$',
                         '$\\frac{3}{2}\\pi$', '2$\\pi$'],
    },
    'FrequencyShift': {
        'label': '\n$\\Delta f \\;$ (Hz)',
        'ticks': UNSCALED_MAG_RANGE * 10,
        'ticks_labels': ['%1.1f' % x for x in UNSCALED_MAG_RANGE * 3],
    },
    'BandstopFilter': {
        'label': '\nbandwidth (Hz)',
        'ticks': UNSCALED_MAG_RANGE * 10,
        'ticks_labels': ['%1.1f' % x for x in UNSCALED_MAG_RANGE * 2],
    },
    'GaussianNoise': {
        'label': '$\\sigma$',
        'ticks': UNSCALED_MAG_RANGE * 10,
        'ticks_labels': ['%1.2f' % x for x in UNSCALED_MAG_RANGE * 0.2],
    },
    'SmoothTimeMask': {
        'label': '$\\Delta t \\;$ (s)',
        'ticks': UNSCALED_MAG_RANGE * 10,
        'ticks_labels': ['%1.1f' % x for x in UNSCALED_MAG_RANGE * 2],
    },
    'ChannelsDropout': {
        'label': '$p_{drop}$',
        'ticks': UNSCALED_MAG_RANGE * 10,
        'ticks_labels': ['%1.1f' % x for x in UNSCALED_MAG_RANGE],
    },
    'ChannelsShuffle': {
        'label': '$p_{shuffle}$',
        'ticks': UNSCALED_MAG_RANGE * 10,
        'ticks_labels': ['%1.1f' % x for x in UNSCALED_MAG_RANGE],
    },
    'SensorsXRotation': {
        'label': '$\\theta_{rot} \\; (^o)$',
        'ticks': UNSCALED_MAG_RANGE * 10,
        'ticks_labels': ['%1.1f' % x for x in UNSCALED_MAG_RANGE * 30],
    },
    'SensorsYRotation': {
        'label': '$\\theta_{rot} \\; (^o)$',
        'ticks': UNSCALED_MAG_RANGE * 10,
        'ticks_labels': ['%1.1f' % x for x in UNSCALED_MAG_RANGE * 30],
    },
    'SensorsZRotation': {
        'label': '$\\theta_{rot} \\; (^o)$',
        'ticks': UNSCALED_MAG_RANGE * 10,
        'ticks_labels': ['%1.1f' % x for x in UNSCALED_MAG_RANGE * 30],
    },
}


def plot_param_search(df, fig_save_path=""):
    # Compute scores
    df_plot = df.query("set == 'valid'").copy()
    df_plot = compute_relative_improvement(df_plot, set='valid',)
    df_plot.score = round(df_plot.score * 100, 2)
    df_plot.augmentation = df_plot.augmentation.apply(
        lambda x: x.replace("()", "")
    )

    aug_names = df_plot.augmentation.sort_values().unique()
    for aug_name in aug_names:
        df_plot = df_plot.append({
            'augmentation': aug_name,
            'magnitude': 0,
            'score': 0,
            'index': 0
        }, ignore_index=True)

    palette = make_fixed_palette()
    _, axes = plt.subplots(1, len(aug_names), sharey=True)

    for ax, aug_name in zip(axes, aug_names):
        sns.pointplot(
            data=df_plot.query("augmentation == @aug_name"),
            x="magnitude",
            y="score",
            hue="augmentation",
            palette=palette,
            ax=ax,
            scale=0.7,
        )
        ax.legend([], [], frameon=False)
        ax.set_xlabel(MAG_RANGES[aug_name]['label'])
        ax.set_ylabel("")
        # ax.set_xticks(ax.get_xticks()[::3])
        ax.set_xticks(MAG_RANGES[aug_name]['ticks'])
        tick_labels = MAG_RANGES[aug_name]['ticks_labels']
        ax.set_xticklabels(tick_labels)
        ax.axhline(y=0, xmin=0, xmax=1, ls='--', c='tab:red')
        ax.set_title(aug_name)
        sns.despine()

    axes[0].set_ylabel(
        'Accuracy relative\nimprovement (%)')
    plt.tight_layout()

    # Save
    if len(fig_save_path) > 0:
        plt.savefig(fig_save_path)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Learning curve plotter",
        "Plotting learnign curve results for Tech Report"
    )

    parser.add_argument(
        "fname",
        type=str,
    )

    parser.add_argument(
        "--tfs-cat",
        type=str,
        default="time",
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
    aug_names = get_tfs_names(df, args.tfs_cat)
    df = df.query('augmentation in @aug_names')

    # Infer task type
    task_properties = infer_task(df)

    # Create saving path
    ext = "png" if args.png else "pdf"
    dataset = task_properties["dataset"]
    fig_dir = Path(eeg_augmentation_benchmark.__file__).parent / (
        f'../outputs/{dataset}/figures/'
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_save_path = fig_dir / \
        f'param-search-{dataset}-{args.tfs_cat}{args.suffix}.{ext}'

    # Plot and save
    plot_param_search(
        df,
        fig_save_path=str(fig_save_path),
    )
