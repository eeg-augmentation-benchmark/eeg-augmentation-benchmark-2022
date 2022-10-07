# Imports
# flake8: noqa F401
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score

import BAE
from BAE.scoring import compute_relative_improvement
from BAE.scoring import compute_score
from plots.plot_utils import setup_style, make_fixed_palette, get_tfs_names
from plots.plot_utils import infer_task


def plot_learning_curve(df, fig_save_path="", lw=0.8):
    # Compute scores
    test_df = df[df['set'] == 'test'].copy()
    test_scores = compute_score(test_df, metric=balanced_accuracy_score)

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

    hue_order = list(test_scores.augmentation.sort_values().unique())
    hue_order.remove("No augmentation")
    hue_order = ["No augmentation"] + hue_order

    # Plot
    palette = make_fixed_palette()
    _, ax = plt.subplots()
    sns.pointplot(
        x='power',
        y='score',
        data=test_scores,
        hue='augmentation',
        hue_order=hue_order,
        linestyles='-',
        dodge=True,
        palette=palette,
        ax=ax,
        lw=lw,
    )
    ax.set_xlabel('Training set proportion')
    ax.set_ylabel('Test set balanced accuracy')
    ax.legend().set_title('')
    plt.tight_layout()

    # Save
    if len(fig_save_path) > 0:
        plt.savefig(fig_save_path)
        plt.show()


# For multi-class

def plot_boxplot(
    test_scores, task_properties, lw=1.2, boxplot_kwargs={}
):
    # rename the x axis to show powers of 2
    test_scores['power'] = test_scores.apply(
        lambda x: '$2^{' + str(int(np.log2(x.proportion))) + '}$',
        axis=1
    )

    proportion = task_properties["proportion"]
    test_scores = test_scores.query(
        "proportion == @proportion"
    ).reset_index(drop=True)

    # Plot
    _, ax = plt.subplots()
    sns.boxplot(
        x='label',
        y='f1_score',
        data=test_scores,
        linewidth=lw,
        ax=ax,
        showfliers=False,
        **boxplot_kwargs
    )
    ax.set_xlabel(task_properties["class_type"])
    ax.set_xticklabels(task_properties["classes_names"])
    ax.legend().set_title('')
    return ax


def plot_rel_boxplot(df, task_properties, fig_save_path="", lw=1.2):
    # Compute scores
    test_df = df[df['set'] == 'test'].copy()

    # Exctract one augmentation
    per_class = list(map(str, task_properties["classes"]))
    test_scores_rel = compute_relative_improvement(
        test_df, per_class=per_class, set='test')

    test_scores_rel.f1_score *= 100

    test_scores_rel.augmentation = test_scores_rel.augmentation.apply(
        lambda x: x.replace("()", "")
    )

    hue_order = list(test_scores_rel.augmentation.sort_values().unique())
    palette = make_fixed_palette()

    ax = plot_boxplot(
        test_scores_rel,
        task_properties,
        lw=1.2,
        boxplot_kwargs={
            "hue": 'augmentation',
            "hue_order": hue_order,
            "palette": palette,
        }
    )
    ax.set_ylabel('F1-score relative\nimprovement (%)')
    ax.axhline(y=0, alpha=0.8, linestyle='--', linewidth=lw, c='tab:red',)
    ax.legend().set_title('')
    if len(hue_order) > 3:
        y1, y2 = ax.get_ylim()
        ax.set_ylim(y1, y2 * 1.5)
        ax.legend(
            title='', loc=0,
            ncol=2,
        )
    plt.tight_layout()

    # Save
    if len(fig_save_path) > 0:
        plt.savefig(fig_save_path)
        plt.show()


def plot_ref_boxplot(df, task_properties, fig_save_path="", lw=1.2):
    # Compute scores
    test_df = df[df['set'] == 'test'].copy()
    ref = test_df.query(
        "augmentation == 'IdentityTransform()'"
    ).reset_index(drop=True)

    # Exctract one augmentation
    per_class = list(map(str, task_properties["classes"]))
    ref_score = compute_score(ref, set='test', per_class=per_class)

    palette = {str(c): "tab:gray" for c in task_properties["classes"]}

    ax = plot_boxplot(
        ref_score,
        task_properties,
        lw=1.2,
        boxplot_kwargs={
            "palette": palette,
        }
    )
    ax.set_ylabel('F1-score')
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
        "--ref",
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
    df = df.query('augmentation in @aug_names').reset_index(drop=True)

    # Infer task type
    task_properties = infer_task(df)

    # Create saving path
    ext = "png" if args.png else "pdf"
    ds = task_properties["dataset"]
    fig_dir = Path(BAE.__file__).parent / f'../outputs/{ds}/figures/'
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not args.ref:
        # Plot and save learning curve
        fig_path_lr = fig_dir / f'LR-{ds}-{args.tfs_cat}{args.suffix}.{ext}'
        plot_learning_curve(
            df,
            fig_save_path=str(fig_path_lr),
        )

        # Plot and save boxplot
        fig_path_bp = fig_dir / \
            f'box-{ds}-{args.tfs_cat}{args.suffix}.{ext}'
        plot_rel_boxplot(
            df,
            task_properties=task_properties,
            fig_save_path=str(fig_path_bp),
        )
    else:
        fig_path_bp = fig_dir / f'box-{ds}-ref-{args.suffix}.{ext}'
        plot_ref_boxplot(
            df,
            task_properties=task_properties,
            fig_save_path=str(fig_path_bp),
        )
