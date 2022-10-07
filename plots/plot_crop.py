# %% Imports
# flake8: noqa F401
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import eeg_augmentation_benchmark
from eeg_augmentation_benchmark.scoring import compute_score
from plot_utils import FONTSIZE
from plot_utils import setup_style

# %% Figure parameters

setup_style()

# %% Load LR-curve (adapt the path)

curves_path = Path(eeg_augmentation_benchmark.__file__).parent / \
    '../outputs/BCI/'
fname_learning_curve = (
    curves_path /
    'Crop_training1.pkl'
)

# %%
df = pd.read_pickle(fname_learning_curve)

classes = np.unique(df.y_true.iloc[0])
if len(classes) == 5:  # sleep scoring
    classes_names = ("W", "N1", "N2", 'N3', "REM")
    dataset = "physionet"
    class_type = 'Sleep stage'
else:
    classes_names = ("foot", "left hand", "right hand", "tongue")
    dataset = "BCI"
    class_type = 'Movement'

df = compute_score(df)

# %%
fig, ax = plt.subplots()
sns.boxplot(
    x='augmentation',
    y='score',
    data=df,
    linewidth=1.2,
    ax=ax,
    showfliers=False)

ax.set_xlabel(class_type, fontsize=FONTSIZE)
ax.set_ylabel('Accuracy', fontsize=FONTSIZE)
y_mean = df.query("augmentation == 'IdentityTransform()'").score.mean()
ax.axhline(
    y=y_mean,
    color='tab:red',
    linestyle='--',
    label='Mean accuracy for ID\n@cc=%1.2f' %
    y_mean)
ax.legend(fontsize=FONTSIZE)
# ax.set_title('Proportion: ' + test_scores_relative_prop['power'].iloc[0],
#              fontsize=FONTSIZE)
# ax.set_ylim(-0.3, 0.9)
plt.tight_layout()
fig_path = Path(eeg_augmentation_benchmark.__file__).parent / f'../outputs/{dataset}/figures/'

plt.savefig(fig_path / f'box_plot_crop.pdf')
plt.show()
