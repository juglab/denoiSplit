# %% [markdown]
"""
# denoiSplit: joint splitting and unsupervised denoising
In this notebook, we tackle the problem of joint splitting and unsupervised denoising, which has a usecase with fluorescence microscopy. From a technical perspective, given a noisy image $x$, the goal is to predict two images $c_1$ and $c_2$ such that $x = c_1 + c_2$. In other words, we have a superimposed image $x$ and we want to predict the denoised estimates of the constituent images $c_1$ and $c_2$. It is important to note that the network is trained with noisy data and the denoising is done in a unsupervised manner. 

For this, we will use [denoiSplit](https://arxiv.org/pdf/2403.11854.pdf), a recently developed approach for this task. In this notebook we train denoiSplit and later evaluate it on one validation frame. The overall schema for denoiSplit is shown below:
<!-- Insert a figure -->
<!-- ![Schema](teaser.png) -->
<img src="teaser.png" alt="drawing" width="800"/>


Here, we look at CCPs (clathrin-coated pits) vs ER (Endoplasmic reticulum) task, one of the tasks tackled by denoiSplit which is generated from [BioSR](https://figshare.com/articles/dataset/BioSR/13264793) dataset. For this task, the noise is synthetically added. 
"""
# %%
# a useful library developed by Google for maintaining the ML configs.
! pip install ml-collections

# %%
! git clone https://github.com/juglab/denoiSplit.git
# %%
import sys
sys.path.append('./denoiSplit')

# %% [markdown]
"""
### Mandatory actions
<div class="alert alert-danger">
1. Set your python kernel to <code>careamics</code> <br>
2. Set the <code>data_dir</code> to the path where the BioSR dataset is present. 
</div>
"""
# %% [markdown]
"""
## Set directories 
In the next cell, we enumerate the necessary fields for this task.
"""
# %%
import os

data_dir = "/group/jug/ashesh/data/BioSR/"  # FILL IN THE PATH TO THE DATA DIRECTORY
work_dir = "."
tensorboard_log_dir = os.path.join(work_dir, "tensorboard_logs")
os.makedirs(tensorboard_log_dir, exist_ok=True)
# %%
import sys

sys.path.append("../../")

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from denoisplit.analysis.plot_utils import clean_ax
from denoisplit.configs.biosr_config import get_config
from denoisplit.training import create_dataset
from denoisplit.nets.model_utils import create_model
from denoisplit.core.metric_monitor import MetricMonitor
from denoisplit.scripts.run import get_mean_std_dict_for_model
from denoisplit.core.data_split_type import DataSplitType
from denoisplit.scripts.evaluate import get_highsnr_data
from denoisplit.analysis.mmse_prediction import get_dset_predictions
from denoisplit.data_loader.patch_index_manager import GridAlignement
from denoisplit.scripts.evaluate import avg_range_inv_psnr, compute_multiscale_ssim

# %% [markdown]
"""
## Config 
<div class="alert alert-block alert-warning"><h3>
    Several Things to try (try some ;) ):</h3>
    <ol>
        <li>Run once with unchanged config to see the performance. </li>
        <li>Increase the noise (double the gaussian noise?) and see how performance degrades. </li>
        <li> Increase the max_epochs, if you want to get better performance. </li>
        <li> For faster training ( but compromising on performance), reduce the number of hierarchy levels and/or the channel count by modifying <em>config.model.z_dims</em>.</li> 
    </ol>
</div>
"""
# %%
# load the default config.
config = get_config()
# Channge the noise level
config.data.poisson_noise_factor = (
    1000  # 1000 is the default value. noise increases with the value.
)
config.data.synthetic_gaussian_scale = (
    5000  # 5000 is the default value. noise increases with the value.
)

# change the number of hierarchy levels.
config.model.z_dims = [128, 128, 128, 128]

# change the training parameters
config.training.lr = 3e-3
config.training.max_epochs = 10
config.training.batch_size = 32
config.training.num_workers = 4

config.workdir = "."
# %% [markdown]
"""
## Create the dataset and pytorch dataloaders. 
"""
# %%
train_dset, val_dset = create_dataset(config, data_dir)
mean_dict, std_dict = get_mean_std_dict_for_model(config, train_dset)
# %%
batch_size = config.training.batch_size
train_dloader = DataLoader(
    train_dset,
    pin_memory=False,
    num_workers=config.training.num_workers,
    shuffle=True,
    batch_size=batch_size,
)
val_dloader = DataLoader(
    val_dset,
    pin_memory=False,
    num_workers=config.training.num_workers,
    shuffle=False,
    batch_size=batch_size,
)
# %% [markdown]
"""
## Create the model.
Here, we instantiate the [denoiSplit model](https://arxiv.org/pdf/2403.11854.pdf). For simplicity, we have disabled the noise model. For enabling the noise model, one would additionally have to train a denoiser. The next step would be to create a noise model using the noisy data and the corresponding denoised predictions. 
"""
# %%
model = create_model(config, mean_dict, std_dict)
model = model.cuda()
# %% [markdown]
"""
## Start training
"""
# %%
# logger = TensorBoardLogger(tensorboard_log_dir, name="", version="", default_hp_metric=False)
logger = None
trainer = pl.Trainer(
    max_epochs=config.training.max_epochs,
    gradient_clip_val=(
        None
        if model.automatic_optimization == False
        else config.training.grad_clip_norm_value
    ),
    logger=logger,
    precision=config.training.precision,
)
trainer.fit(model, train_dloader, val_dloader)
# %% [markdown]
"""
## Evaluate the model
"""
# %%
model.eval()
_ = model.cuda()
eval_frame_idx = 0
# reducing the data, just for speed
val_dset.reduce_data(t_list=[eval_frame_idx])
mmse_count = 10
overlapping_padding_kwargs = {
    "mode": config.data.get("padding_mode", "constant"),
}
if overlapping_padding_kwargs["mode"] == "constant":
    overlapping_padding_kwargs["constant_values"] = config.data.get("padding_value", 0)
val_dset.set_img_sz(
    128,
    32,
    grid_alignment=GridAlignement.Center,
    overlapping_padding_kwargs=overlapping_padding_kwargs,
)

# MMSE prediction
pred_tiled, rec_loss, logvar_tiled, patch_psnr_tuple, pred_std_tiled = (
    get_dset_predictions(
        model,
        val_dset,
        batch_size,
        num_workers=config.training.num_workers,
        mmse_count=mmse_count,
        model_type=config.model.model_type,
    )
)

# One sample prediction
pred1_tiled, *_ = get_dset_predictions(
    model,
    val_dset,
    batch_size,
    num_workers=config.training.num_workers,
    mmse_count=1,
    model_type=config.model.model_type,
)
# One sample prediction
pred2_tiled, *_ = get_dset_predictions(
    model,
    val_dset,
    batch_size,
    num_workers=config.training.num_workers,
    mmse_count=1,
    model_type=config.model.model_type,
)
# %% [markdown]
"""
## Stich the predictions
"""
# %%
from denoisplit.analysis.stitch_prediction import stitch_predictions

pred = stitch_predictions(pred_tiled, val_dset)


# ignore pixels at the [right/bottom] boundary.
def print_ignored_pixels():
    ignored_pixels = 1
    while (
        pred[
            0,
            -ignored_pixels:,
            -ignored_pixels:,
        ].std()
        == 0
    ):
        ignored_pixels += 1
    ignored_pixels -= 1
    return ignored_pixels


actual_ignored_pixels = print_ignored_pixels()
pred = pred[:, :-actual_ignored_pixels, :-actual_ignored_pixels]
pred1 = stitch_predictions(pred1_tiled, val_dset)[
    :, :-actual_ignored_pixels, :-actual_ignored_pixels
]
pred2 = stitch_predictions(pred2_tiled, val_dset)[
    :, :-actual_ignored_pixels, :-actual_ignored_pixels
]
# %%
highres_data = get_highsnr_data(config, data_dir, DataSplitType.Val)

highres_data = highres_data[
    eval_frame_idx : eval_frame_idx + 1,
    :-actual_ignored_pixels,
    :-actual_ignored_pixels,
]

noisy_data = val_dset._noise_data[..., 1:] + val_dset._data
noisy_data = noisy_data[..., :-actual_ignored_pixels, :-actual_ignored_pixels, :]
model_input = np.mean(noisy_data, axis=-1)
# %% [markdown]
"""
# Qualitative performance on a random crop
denoiSplit is capable of sampling from a learned posterior.
Here we show full input frame and a randomly cropped input (300*300),
two corresponding prediction samples, the difference between the two samples (S1−S2),
the MMSE prediction, and otherwise unused high SNR microscopy crop. 
The MMSE predictions are computed by averaging 10 samples. 
"""


# %%
def add_str(ax_, txt):
    """
    Add psnr string to the axes
    """
    textstr = txt
    props = dict(boxstyle="round", facecolor="gray", alpha=0.5)
    # place a text box in upper left in axes coords
    ax_.text(
        0.05,
        0.95,
        textstr,
        transform=ax_.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
        color="white",
    )


ncols = 7
nrows = 2
sz = 300
hs = np.random.randint(0, highres_data.shape[1] - sz)
ws = np.random.randint(0, highres_data.shape[2] - sz)
_, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
ax[0, 0].imshow(model_input[0], cmap="magma")

rect = patches.Rectangle((ws, hs), sz, sz, linewidth=1, edgecolor="r", facecolor="none")
ax[0, 0].add_patch(rect)
ax[1, 0].imshow(model_input[0, hs : hs + sz, ws : ws + sz], cmap="magma")
add_str(ax[0, 0], "Full Input Frame")
add_str(ax[1, 0], "Random Input Crop")

ax[0, 1].imshow(noisy_data[0, hs : hs + sz, ws : ws + sz, 0], cmap="magma")
ax[1, 1].imshow(noisy_data[0, hs : hs + sz, ws : ws + sz, 1], cmap="magma")

ax[0, 2].imshow(pred1[0, hs : hs + sz, ws : ws + sz, 0], cmap="magma")
ax[1, 2].imshow(pred1[0, hs : hs + sz, ws : ws + sz, 1], cmap="magma")

ax[0, 3].imshow(pred2[0, hs : hs + sz, ws : ws + sz, 0], cmap="magma")
ax[1, 3].imshow(pred2[0, hs : hs + sz, ws : ws + sz, 1], cmap="magma")

diff = pred2 - pred1
ax[0, 4].imshow(diff[0, hs : hs + sz, ws : ws + sz, 0], cmap="coolwarm")
ax[1, 4].imshow(diff[0, hs : hs + sz, ws : ws + sz, 1], cmap="coolwarm")

ax[0, 5].imshow(pred[0, hs : hs + sz, ws : ws + sz, 0], cmap="magma")
ax[1, 5].imshow(pred[0, hs : hs + sz, ws : ws + sz, 1], cmap="magma")


ax[0, 6].imshow(highres_data[0, hs : hs + sz, ws : ws + sz, 0], cmap="magma")
ax[1, 6].imshow(highres_data[0, hs : hs + sz, ws : ws + sz, 1], cmap="magma")
plt.subplots_adjust(wspace=0.02, hspace=0.02)
ax[0, 0].set_title("Model Input", size=13)
ax[0, 1].set_title("Target", size=13)
ax[0, 2].set_title("Sample 1 (S1)", size=13)
ax[0, 3].set_title("Sample 2 (S2)", size=13)
ax[0, 4].set_title('"S2" - "S1"', size=13)
ax[0, 5].set_title(f"Prediction MMSE({mmse_count})", size=13)
ax[0, 6].set_title("High SNR Reality", size=13)

twinx = ax[0, 6].twinx()
twinx.set_ylabel("Channel 1", size=13)
clean_ax(twinx)
twinx = ax[1, 6].twinx()
twinx.set_ylabel("Channel 2", size=13)
clean_ax(twinx)
clean_ax(ax)

# %% [markdown]
"""
# Qualitative performance on multiple random crops
"""
# %%
nimgs = 3
ncols = 7
nrows = 2 * nimgs
sz = 300
_, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))

for img_idx in range(nimgs):
    hs = np.random.randint(0, highres_data.shape[1] - sz)
    ws = np.random.randint(0, highres_data.shape[2] - sz)
    ax[2 * img_idx, 0].imshow(model_input[0], cmap="magma")

    rect = patches.Rectangle(
        (ws, hs), sz, sz, linewidth=1, edgecolor="r", facecolor="none"
    )
    ax[2 * img_idx, 0].add_patch(rect)
    ax[2 * img_idx + 1, 0].imshow(
        model_input[0, hs : hs + sz, ws : ws + sz], cmap="magma"
    )
    add_str(ax[2 * img_idx, 0], "Full Input Frame")
    add_str(ax[2 * img_idx + 1, 0], "Random Input Crop")

    ax[2 * img_idx, 1].imshow(
        noisy_data[0, hs : hs + sz, ws : ws + sz, 0], cmap="magma"
    )
    ax[2 * img_idx + 1, 1].imshow(
        noisy_data[0, hs : hs + sz, ws : ws + sz, 1], cmap="magma"
    )

    ax[2 * img_idx, 2].imshow(pred1[0, hs : hs + sz, ws : ws + sz, 0], cmap="magma")
    ax[2 * img_idx + 1, 2].imshow(pred1[0, hs : hs + sz, ws : ws + sz, 1], cmap="magma")

    ax[2 * img_idx, 3].imshow(pred2[0, hs : hs + sz, ws : ws + sz, 0], cmap="magma")
    ax[2 * img_idx + 1, 3].imshow(pred2[0, hs : hs + sz, ws : ws + sz, 1], cmap="magma")

    diff = pred2 - pred1
    ax[2 * img_idx, 4].imshow(diff[0, hs : hs + sz, ws : ws + sz, 0], cmap="coolwarm")
    ax[2 * img_idx + 1, 4].imshow(
        diff[0, hs : hs + sz, ws : ws + sz, 1], cmap="coolwarm"
    )

    ax[2 * img_idx, 5].imshow(pred[0, hs : hs + sz, ws : ws + sz, 0], cmap="magma")
    ax[2 * img_idx + 1, 5].imshow(pred[0, hs : hs + sz, ws : ws + sz, 1], cmap="magma")

    ax[2 * img_idx, 6].imshow(
        highres_data[0, hs : hs + sz, ws : ws + sz, 0], cmap="magma"
    )
    ax[2 * img_idx + 1, 6].imshow(
        highres_data[0, hs : hs + sz, ws : ws + sz, 1], cmap="magma"
    )

    twinx = ax[2 * img_idx, 6].twinx()
    twinx.set_ylabel("Channel 1", size=15)
    clean_ax(twinx)

    twinx = ax[2 * img_idx + 1, 6].twinx()
    twinx.set_ylabel("Channel 2", size=15)
    clean_ax(twinx)

ax[0, 0].set_title("Model Input", size=15)
ax[0, 1].set_title("Target", size=15)
ax[0, 2].set_title("Sample 1 (S1)", size=15)
ax[0, 3].set_title("Sample 2 (S2)", size=15)
ax[0, 4].set_title('"S2" - "S1"', size=15)
ax[0, 5].set_title(f"Prediction MMSE({mmse_count})", size=15)
ax[0, 6].set_title("High SNR Reality", size=15)

clean_ax(ax)
plt.subplots_adjust(wspace=0.02, hspace=0.02)
# plt.tight_layout()
# %% [markdown]
"""
## Quantitative performance
We evaluate on two metrics, Multiscale SSIM and PSNR.
"""
# %%
mean_tar = mean_dict["target"].cpu().numpy().squeeze().reshape(1, 1, 1, 2)
std_tar = std_dict["target"].cpu().numpy().squeeze().reshape(1, 1, 1, 2)
pred_unnorm = pred * std_tar + mean_tar

psnr_list = [
    avg_range_inv_psnr(highres_data[..., i].copy(), pred_unnorm[..., i].copy())
    for i in range(highres_data.shape[-1])
]
ssim_list = compute_multiscale_ssim(highres_data.copy(), pred_unnorm.copy())
print("Metric: Ch1\t Ch2")
print(f"PSNR  : {psnr_list[0]:.2f}\t {psnr_list[1]:.2f}")
print(f"MS-SSIM  : {ssim_list[0]:.3f}\t {ssim_list[1]:.3f}")
