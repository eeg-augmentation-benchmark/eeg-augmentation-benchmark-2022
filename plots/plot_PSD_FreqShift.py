# %% imports

from pathlib import Path
import matplotlib.pyplot as plt
import torch
from braindecode.augmentation import FrequencyShift

from plot_utils import FONTSIZE
from plot_utils import W
from plot_utils import plot_psd
from plot_utils import setup_style
from plot_utils import get_windows

# %load_ext autoreload
# %autoreload 2

# %% Figure parameters

setup_style()

# %%
sfreq = 100
# This allows to apply the transform with a fixed shift (10 Hz) for
# visualization instead of sampling the shift randomly between -2 and 2 Hz
transform = FrequencyShift(
    probability=1,
    sfreq=100,
    max_delta_freq=2,
    random_state=19
)
delta = 0.7
# %%

fig, ax = plt.subplots(figsize=(W, W * 9 / 16 * 2 / 3))
# fig, ax = plt.subplots()

# subject 2
windows = get_windows()
epochs = windows.datasets[6].windows['Sleep stage 2']  # original epochs
X = epochs.get_data()
epochs_2 = windows.datasets[16].windows['Sleep stage 2']
X_2 = epochs_2.get_data()
X_tr, _ = transform.operation(torch.as_tensor(X_2).float(), None, delta, sfreq)
plot_psd(
    X_2,
    ax,
    label='Subject 1',
    c='tab:olive',
    fmin=0, fmax=20
)
plot_psd(
    X,
    ax,
    label='Subject 2',
    c='tab:blue',
    fmin=0, fmax=20
)
plot_psd(
    X_tr.numpy(),
    ax,
    label='Subject 1 shifted\n$\\Delta f = {}$ Hz'.format(delta),
    c='tab:red',
    ls='-.',
    fmin=0, fmax=20
)

ax.set_ylabel(ax.get_ylabel(), fontsize=FONTSIZE)
ax.set_xlabel(ax.get_xlabel(), fontsize=FONTSIZE)
ax.set_title('')
ax.set_ylim(ymin=-10)
ax.axvline(x=0, ymin=0, ymax=1e3, ls='--', c='k', alpha=0.8, zorder=0)

ax.legend(
    fontsize=FONTSIZE,
    ncol=2,
    borderaxespad=0,
    frameon=False,
    loc='center',
    bbox_to_anchor=(0.65, 0.7)
)
fig.tight_layout()
fig_dir = Path(__file__).parent / '..' / 'outputs/physionet/figures/'
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / 'PSD_FreqShift.pdf')
plt.savefig(fig_dir / 'PSD_FreqShift.png')
plt.show()

# %%
