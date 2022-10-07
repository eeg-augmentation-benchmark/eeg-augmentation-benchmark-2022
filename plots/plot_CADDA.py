# %%

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, f1_score

plt.style.use('seaborn-paper')
plt.tight_layout()
FONTSIZE = 15
plt.rcParams["figure.figsize"] = (11, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.grid.axis"] = "y"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
plt.rcParams['font.size'] = FONTSIZE

# %%


def box_score(
        df,
        per_class,
        metric,
        n_classes=5,
        set='test',
        aug_name=None,
        ref=None,
        n_folds=None):
    """
    Compute the class-wise or multi-class score based on the columns y_pred
    and y_true of a DataFrame.

    Parameters:
    -----------
    df: pd.DataFrame
        DataFrame which contains the predictions and ground truth.
    per_class: bool
        Whether to compute a class-wise score or a multi-class score.
    metric: function
        metric to use in order to compute the score.
    n_classes: int
        Number of classes. Only needed for class-wise score.
    set: str
        Set to compute the score on (train, valid or test).
    aug_name: str
        Name of the augmentation.
    ref: pd.DataFrame, optional
        If parsed the box score will return the difference between the
        computed score and the reference score.

    Returns:
    --------
    pd.DataFrame
        DataFrame that contains the columns
    """

    df_test = df[df['set'] == set]
#    if n_folds:
#        df_test = df_test[df_test['fold'] <= 10]

    output = []
    if per_class:
        for c in range(n_classes):
            for id, y_pred, y_true in zip(
                    df_test.index, df_test.y_pred, df_test.y_true):
                output.append({
                    'class': c,
                    'score': metric(
                        y_pred=y_pred == c,
                        y_true=y_true == c),
                    'fold': df_test.fold[id]})
    else:
        for id, y_pred, y_true in zip(
                df_test.index, df_test.y_pred, df_test.y_true):
            output.append({
                'score': metric(y_pred, y_true),
                'fold': df_test.fold[id]})
    output = pd.DataFrame(output)
    if not aug_name:
        aug_name = str(np.array(df_test.augmentation)[0]).split('(')[0]
    output['augmentation'] = aug_name
    df = pd.DataFrame(output)

    if isinstance(ref, pd.DataFrame):
        # Make sure that each fold is compared to its counterpart.
        assert np.array_equal(df.fold, ref.fold)
        df.score = (df.score - ref.score) / ref.score
    return df


def box_plot(score, ax, per_class=True, save_fig=None):
    """
    Make a box-plot figure based on a DataFrame that contains the scores. The
    score DataFrame can be computed using the box_score function. The DataFrame
    might contain the scores of different augmentations, the plot will use
    hue=augmentation.

    Parameters:
    -----------
    score: pd.DataFrame
        The DataFrame which contains the scores and the augmentations name.
    ax: plt.axes
    per_class: bool
        Whether to use the per_class display or not.

    """
    if per_class:
        sns.boxplot(x='class',
                    y="score",
                    hue="augmentation",
                    data=score,
                    ax=ax)
        ax.set_xlabel('')
        ax.set_xticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
        ax.set_title('')
        ax.set_ylabel(
            '$\\mathrm{F1-score}$ relative improvement',
            fontsize=FONTSIZE)
        plt.title(
            "Per class $\\mathrm{F1-score}$ using 250 windows for training",
            fontsize=FONTSIZE)
        plt.tight_layout()

    else:
        sns.boxplot(x='augmentation',
                    y="score",
                    data=score,)
        ax.set_ylabel(
            '$\\Delta_{\\mathrm{balanced\\;accuracy}}$',
            fontsize=FONTSIZE + 8)
        ax.set_xlabel('')
        if len(score) > 50:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        plt.title(
            'Multi class balanced accuracy score using 250 windows for'
            'training', fontsize=FONTSIZE)
        plt.tight_layout()
    plt.legend(loc=0, ncol=3, fontsize=FONTSIZE)
    if save_fig:
        plt.savefig(save_fig)
    plt.tight_layout()
    return ax


# %% PLot all the .pkl files of a folder.
os.chdir('/home/user/Documents/BAE')
hist_path = Path('./outputs/hist/drago5/no_rng/')
PER_CLASS = True
METRIC = f1_score

###
# Compute the ref score
###
ref = pd.read_pickle(hist_path / 'LR-IdentityTransform-10-17-38-p0.0055.pkl')
ref_score = box_score(
    df=ref,
    per_class=PER_CLASS,
    metric=METRIC,
    aug_name='Identity',)

###
# Compute the score of each augmentation
###
aug_1 = 'GaussianNoise'
aug_2 = 'SignFlip'
aug_3 = 'custom_2'
scores_list = []
for file in hist_path.iterdir():
    df = pd.read_pickle(file)
# assert not df.drop(axis=1,columns='fold_random_state').isnull().values.any()
    score = box_score(
        df=df,
        per_class=PER_CLASS,
        metric=METRIC,
        ref=ref_score,
        aug_name=str(file).split('-')[1],)
    if score.augmentation[1] in [aug_1, aug_2]:
        print(df.augmentation[1])
        scores_list.append(score)
df_scores = pd.concat(scores_list)
assert not df_scores.isnull().values.any()

###
# Remove the Identity
###
df_scores = pd.concat(scores_list)
df_scores.index = np.arange(len(df_scores))
df_scores.drop(df_scores[df_scores['augmentation'] ==
               'IdentityTransform'].index, inplace=True)
###
# Plot !
###
fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})


# %% Class-wise
box_plot(
    score=df_scores,
    ax=axes[0],
    per_class=PER_CLASS)
axes[0].set_title('(a)', fontsize=FONTSIZE)
axes[0].set_ylim(ymin=-0.6, ymax=0.9)
axes[0].axhline(y=0, alpha=0.8, linestyle='--', linewidth=1.2, c='tab:red',)

h, _ = axes[0].get_legend_handles_labels()
axes[0].legend(h[:2],
               ["$\\mathtt{Gaussian \\;noise}$",
                "$\\mathtt{sign\\; flip}$"],
               loc=0,
               fontsize=FONTSIZE)
# %%
PER_CLASS_2 = False
METRIC_2 = balanced_accuracy_score

ref_score_2 = box_score(
    df=ref,
    per_class=PER_CLASS_2,
    metric=METRIC_2,
    aug_name='Identity',)

scores_list_2 = []
for file in hist_path.iterdir():
    df = pd.read_pickle(file)
    score = box_score(
        df=df,
        per_class=PER_CLASS_2,
        metric=METRIC_2,
        ref=ref_score_2,
        aug_name=str(file).split('-')[1],)
    if score.augmentation[1] in [aug_1, aug_2, aug_3]:
        scores_list_2.append(score)
df_scores_2 = pd.concat(scores_list_2)
assert not df_scores_2.isnull().values.any()

diff_1 = df_scores_2[df_scores_2['augmentation'] == aug_3].score - \
    df_scores_2[df_scores_2['augmentation'] == aug_1].score
diff_2 = df_scores_2[df_scores_2['augmentation'] == aug_3].score - \
    df_scores_2[df_scores_2['augmentation'] == aug_2].score

diff_aug_1 = pd.DataFrame({
    'difference': diff_1 / ref_score_2.score, })
diff_aug_1['augmentation'] = aug_1

diff_aug_2 = pd.DataFrame({
    'difference': diff_2 / ref_score_2.score})

diff_aug_2['augmentation'] = aug_2
diff = pd.concat([diff_aug_1, diff_aug_2])

# %%

sns.boxplot(
    data=diff,
    x='augmentation',
    y='difference',
    ax=axes[1],
    width=0.3,
    notch=1,
)
axes[1].set_title(
    '(b)',
    fontsize=FONTSIZE)
axes[1].axhline(y=0, alpha=0.8, linestyle='--', linewidth=1.2, c='tab:red',)
axes[1].set_xticklabels([
    'CW$-$\n $ \\mathtt{Gaussian \\; noise}$',
    'CW$-$\n $ \\mathtt{sign \\; flip}$'
], fontsize=FONTSIZE - 3)
axes[1].set_ylabel(
    'multi-class $\\mathrm{balanced\\;accuracy}$\nrelative improvement',
    fontsize=FONTSIZE)
axes[1].set_xlabel('')
axes[1].set_ylim(ymin=-0.10, ymax=0.2)

plt.tight_layout()
plt.savefig('./outputs/hist/figures/class-wise-aug')
plt.show()
