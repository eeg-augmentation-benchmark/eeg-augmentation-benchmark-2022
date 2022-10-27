# %% imports
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from braindecode.augmentation import FTSurrogate

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
N3_indices = np.where(labels == 2)
# %%
index = N3_indices[0][21]
window_test = windows[index][0][0]

# %%

t_start, t_stop = 200, 1500

transform = FTSurrogate(
    probability=1,
    phase_noise_magnitude=1,
    random_state=13  # 13 12 16 14
)
window_flip = transform(window_test)

# %%
fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
t_start, t_stop = 800, 1400

plot_signal(
    window_test,
    ax=axes[0],
    t_start=t_start,
    t_stop=t_stop,
    alpha=0.6,
    c='k',
    label='Original Signal',
)
plot_signal(
    window_test,
    ax=axes[0],
    t_start=985,
    t_stop=1100,
    alpha=1,
    c='tab:red',
    linestyle='-',
    label='K-complex',
)
plot_signal(
    window_flip,
    ax=axes[1],
    t_start=t_start,
    t_stop=t_stop,
    alpha=0.8,
    c='tab:blue',
    label='Surrogate Signal',
    ls='-'
)
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.set_yticklabels([])
    ax.margins(x=0)

axes[0].legend(fontsize=FONTSIZE, ncol=2, loc='upper right',
               bbox_to_anchor=(1, 1.1), frameon=True)
axes[1].legend(fontsize=FONTSIZE, ncol=2, loc='upper right',
               bbox_to_anchor=(1, 1.1), frameon=True)
axes[1].set_xlabel('Time (s)', fontsize=FONTSIZE)
fig.tight_layout()
fig_dir = Path(__file__).parent / '..' / 'outputs/physionet/figures/'
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "FTSurrogate_K.pdf")
plt.savefig(fig_dir / "FTSurrogate_K.png")
plt.show()

# %%
