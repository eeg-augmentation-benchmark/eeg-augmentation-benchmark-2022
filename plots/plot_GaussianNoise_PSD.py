# %% imports
from pathlib import Path
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from braindecode.augmentation.transforms import GaussianNoise
from matplotlib import cm

from plot_utils import plot_psd
from plot_utils import setup_style, get_windows

# %load_ext autoreload
# %autoreload 2

# %% Figure parameters

setup_style()

# %%
windows = get_windows()
epochs = windows.datasets[3].windows['Sleep stage 1']
X = epochs.get_data()

fig, ax = plt.subplots()
params = np.linspace(0.001, 0.1, 30) * 5
colors = cm.get_cmap('viridis', len(params)).colors

for std, color in zip(params, colors):
    transform = GaussianNoise(
        probability=1,
        std=std
    )
    X_T = transform(copy(X))
    plot_psd(
        X_T,
        ax=ax,
        fmin=3,
        fmax=35,
        c=color,
        label='SNR$={:.0e}$'.format(std)
    )

# setup the colorbar
scalarmappaple = cm.ScalarMappable(cmap=cm.get_cmap('viridis', len(params)))
scalarmappaple.set_array(params)
cbar = plt.colorbar(scalarmappaple)
cbar.set_label('$\\sigma$')
ax.set_ylabel(ax.get_ylabel())
ax.set_xlabel(ax.get_xlabel())
plt.tight_layout()
fig_dir = Path(__file__).parent / '..' / 'outputs/physionet/figures/'
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / 'PSD_GaussianNoise.pdf')
plt.savefig(fig_dir / 'PSD_GaussianNoise.png')
plt.show()

# %%
