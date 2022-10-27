# %% imports
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from braindecode.augmentation import SignFlip

from eeg_augmentation_benchmark.utils import get_labels
from plot_utils import FONTSIZE
from plot_utils import plot_signal
from plot_utils import setup_style
from plot_utils import get_windows

# %load_ext autoreload
# %autoreload 2

# %% Figure parameters

setup_style()

# %%
windows = get_windows()
labels = get_labels(windows)
N2_indices = np.where(labels == 0)
index = N2_indices[0][26]
window_test = windows[index][0][0]


t_start, t_stop = 500, 1300
time_reverse = SignFlip(probability=1)
window_flip = time_reverse(window_test)

# %%
fig, axes = plt.subplots(nrows=2, sharex=True)

plot_signal(
    window_test,
    ax=axes[0],
    t_start=3000 - t_stop,
    t_stop=3000 - t_start,
    alpha=0.6,
    c='k',
    label='Original Signal',

)
plot_signal(
    window_flip,
    ax=axes[1],
    t_start=3000 - t_stop,
    t_stop=3000 - t_start,
    alpha=0.8,
    c='tab:blue',
    label='Flipped Signal',
    ls='-'
)
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['left'].set_visible(True)
    ax.margins(x=0, y=0.3)

axes[0].legend(
    fontsize=FONTSIZE, bbox_to_anchor=(0.25, 0.7), loc="lower center",
    frameon=False
)
axes[1].legend(
    fontsize=FONTSIZE, bbox_to_anchor=(0.25, -0.05), loc="lower center",
    frameon=False
)

# axes[1].set_xticklabels(np.array(axes[1].get_xticks(), dtype=int) / 100)
axes[1].set_xlabel('Time (s)', fontsize=FONTSIZE)
fig.tight_layout()
fig_dir = Path(__file__).parent / '..' / 'outputs/physionet/figures/'
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / 'sign_flip.pdf')
plt.savefig(fig_dir / 'sign_flip.png')
# %%

map_stage = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
subject = windows.split('subject')['1']
labels_1 = get_labels(subject)
stages = [map_stage[k] for k in labels_1]
fig, ax = plt.subplots()
ax.plot(stages[30:400])
ax.set_xlabel('Time (min)', fontsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('Sleep stage', fontsize=FONTSIZE)
plt.yticks(
    ticks=[0, 1, 2, 3, 4],
    labels=['N3', 'N2', 'N1', 'REM', 'Wake'],
    fontsize=FONTSIZE)
plt.tight_layout()
plt.savefig(fig_dir / 'sleep_staging.pdf')
plt.savefig(fig_dir / 'sleep_staging.png')
plt.show()
# %%
