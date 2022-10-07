# %% imports
import matplotlib.pyplot as plt
import torch
from braindecode.augmentation import SmoothTimeMask

from plot_utils import FONTSIZE
from plot_utils import W
from plot_utils import plot_signal
from plot_utils import setup_style
from plot_utils import get_windows

# %load_ext autoreload
# %autoreload 2

# %% Figure parameters

setup_style()

# %%
windows = get_windows()
window_test = windows[19][0][0]

# %%

t_start, t_stop = 200, 1500

window_masked = SmoothTimeMask.operation(
    torch.Tensor(window_test).unsqueeze(0).unsqueeze(0),
    y=torch.Tensor([1]),
    mask_start_per_sample=torch.Tensor([1880]),
    mask_len_samples=160,
)[0][0, 0]
# %%
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(W, W / 2.25))
t_start, t_stop = 1600, 2200

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
    window_masked,
    ax=axes[1],
    t_start=t_start,
    t_stop=t_stop,
    alpha=0.8,
    c='tab:blue',
    label='Masked Signal',
    ls='-'
)
# plot_signal(
#     window_masked,
#     ax=axes[1],
#     t_start=1880,
#     t_stop=2040,
#     alpha=1,
#     c='tab:red',
#     label='',
#     ls='-'
# )

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.set_yticklabels([])

# axes[1].set_xticklabels(np.array(axes[1].get_xticks(), dtype=int) / 100)
axes[1].set_xlabel('Time (s)', fontsize=FONTSIZE)
axes[0].legend(fontsize=FONTSIZE, ncol=2, loc='center',
               bbox_to_anchor=(0.16, 0.9), frameon=True)
axes[1].legend(fontsize=FONTSIZE, ncol=3, loc='center',
               bbox_to_anchor=(0.16, 0.9), frameon=True)
fig.tight_layout()
plt.savefig("../outputs/physionet/figures/TimeMask.pdf")
plt.savefig("../outputs/physionet/figures/TimeMask.png")
plt.show()
# %%
