# %% imports
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from braindecode.augmentation.transforms import TimeReverse

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
N2_indices = np.where(get_labels(windows) == 2)[0]

# %%
idx = N2_indices[6]
window_test = windows[idx][0][0]
fig, axes = plt.subplots(nrows=2, sharex=True)
t_start, t_stop = 1600, 2200
plot_signal(
    window_test,
    ax=axes[0],
    t_start=t_start,
    t_stop=t_stop,
    alpha=1,
    c='tab:blue',
    label='Original Signal',
)
# %%

window_reversed = TimeReverse.operation(
    torch.Tensor(window_test).unsqueeze(0).unsqueeze(0),
    y=torch.Tensor([1]),
)[0][0, 0]
# %%
fig, axes = plt.subplots(nrows=2, sharex=False)
t_start, t_stop = 1600, 2000
K_start, K_stop = 1800, 1850
plot_signal(
    window_test,
    ax=axes[0],
    t_start=t_start,
    t_stop=t_stop,
    alpha=0.6,
    c='k',
    label='Original signal',
)
plot_signal(
    window_test,
    ax=axes[0],
    t_start=K_start,
    t_stop=K_stop,
    alpha=1,
    c='tab:red',
    label='K-complex',
)
plot_signal(
    window_reversed,
    ax=axes[1],
    t_start=3000 - t_stop,
    t_stop=3000 - t_start,
    alpha=0.8,
    c='tab:blue',
    label='Reversed signal',
    ls='-'
)
plot_signal(
    window_reversed,
    ax=axes[1],
    t_start=3000 - K_stop,
    t_stop=3000 - K_start,
    alpha=1,
    c='tab:red',
    # label='Reversed K-complex',
    ls='-'
)
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.margins(x=0, y=0.3)

# axes[1].set_xticklabels(np.array(axes[1].get_xticks(), dtype=int)[::] / 100)
axes[1].set_xlabel('Time (s)', fontsize=FONTSIZE)
axes[0].legend(fontsize=FONTSIZE, ncol=2, loc='upper center',
               bbox_to_anchor=(0.3, 0.35), frameon=True)
axes[1].legend(fontsize=FONTSIZE, ncol=3, loc='upper center',
               bbox_to_anchor=(0.2, 0.35), frameon=True)
fig.tight_layout()
fig_dir = Path(__file__).parent / '..' / 'outputs/physionet/figures/'
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "time_reverse_K.pdf")
plt.savefig(fig_dir / "time_reverse_K.png")
plt.show()
# %%
