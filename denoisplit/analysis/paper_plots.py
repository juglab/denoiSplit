import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from denoisplit.analysis.plot_utils import add_left_arrow, add_pixel_kde, add_right_arrow, clean_ax
from denoisplit.core.psnr import RangeInvariantPsnr


def get_plotoutput_dir(ckpt_dir, patch_size, mmse_count=50):
    plotsrootdir = f'/group/jug/ashesh/data/paper_figures/patch_{patch_size}_mmse_{mmse_count}'
    rdate, rconfig, rid = ckpt_dir.split("/")[-3:]
    fname_prefix = rdate + '-' + rconfig.replace('-', '')[:-2] + '-' + rid
    plotsdir = os.path.join(plotsrootdir, fname_prefix)
    os.makedirs(plotsdir, exist_ok=True)
    print(plotsdir)
    return plotsdir


def get_last_index(bin_count, quantile):
    cumsum = np.cumsum(bin_count)
    normalized_cumsum = cumsum / cumsum[-1]
    for i in range(1, len(normalized_cumsum)):
        if normalized_cumsum[-i] < quantile:
            return i - 1
    return None


def get_first_index(bin_count, quantile):
    cumsum = np.cumsum(bin_count)
    normalized_cumsum = cumsum / cumsum[-1]
    for i in range(len(normalized_cumsum)):
        if normalized_cumsum[i] > quantile:
            return i
    return None


def plot_calibration(ax, calibration_stats):
    first_idx = get_first_index(calibration_stats[0]['bin_count'], 0.001)
    last_idx = get_last_index(calibration_stats[0]['bin_count'], 0.999)
    ax.plot(calibration_stats[0]['rmv'][first_idx:-last_idx],
            calibration_stats[0]['rmse'][first_idx:-last_idx],
            'o',
            label='$\hat{C}_0$')

    first_idx = get_first_index(calibration_stats[1]['bin_count'], 0.001)
    last_idx = get_last_index(calibration_stats[1]['bin_count'], 0.999)
    ax.plot(calibration_stats[1]['rmv'][first_idx:-last_idx],
            calibration_stats[1]['rmse'][first_idx:-last_idx],
            'o',
            label='$\hat{C}_1$')

    ax.set_xlabel('RMV')
    ax.set_ylabel('RMSE')
    ax.legend()


# add_left_arrow(axes_list[3], (155,80), arrow_length=50)
def get_psnr_str(tar_hsnr, pred, col_idx):
    return f'{RangeInvariantPsnr(tar_hsnr[col_idx][None], pred[col_idx][None]).item():.1f}'


def add_psnr_str(ax_, psnr):
    """
    Add psnr string to the axes
    """
    textstr = f'PSNR\n{psnr}'
    props = dict(boxstyle='round', facecolor='gray', alpha=0.5)
    # place a text box in upper left in axes coords
    ax_.text(0.05,
             0.95,
             textstr,
             transform=ax_.transAxes,
             fontsize=11,
             verticalalignment='top',
             bbox=props,
             color='white')


def get_predictions(idx, val_dset, model, mmse_count=50, patch_size=256):
    print(f'Predicting for {idx}')
    val_dset.set_img_sz(patch_size, 64)

    with torch.no_grad():
        # val_dset.enable_noise()
        inp, tar = val_dset[idx]
        # val_dset.disable_noise()

        inp = torch.Tensor(inp[None])
        tar = torch.Tensor(tar[None])
        inp = inp.cuda()
        x_normalized = model.normalize_input(inp)
        tar = tar.cuda()
        tar_normalized = model.normalize_target(tar)

        recon_img_list = []
        for _ in range(mmse_count):
            recon_normalized, td_data = model(x_normalized)
            rec_loss, imgs = model.get_reconstruction_loss(recon_normalized,
                                                           x_normalized,
                                                           tar_normalized,
                                                           return_predicted_img=True)
            imgs = model.unnormalize_target(imgs)
            recon_img_list.append(imgs.cpu().numpy()[0])

    recon_img_list = np.array(recon_img_list)
    return inp, tar, recon_img_list


def show_for_one(idx,
                 val_dset,
                 highsnr_val_dset,
                 model,
                 calibration_stats,
                 mmse_count=5,
                 patch_size=256,
                 num_samples=2,
                 baseline_preds=None):
    highsnr_val_dset.set_img_sz(patch_size, 64)
    highsnr_val_dset.disable_noise()
    _, tar_hsnr = highsnr_val_dset[idx]
    inp, tar, recon_img_list = get_predictions(idx, val_dset, model, mmse_count=mmse_count, patch_size=patch_size)
    plot_crops(inp,
               tar,
               tar_hsnr,
               recon_img_list,
               calibration_stats,
               num_samples=num_samples,
               baseline_preds=baseline_preds)


def plot_crops(inp, tar, tar_hsnr, recon_img_list, calibration_stats, num_samples=2, baseline_preds=None):
    if baseline_preds is None:
        baseline_preds = []
    if len(baseline_preds) > 0:
        for i in range(len(baseline_preds)):
            if baseline_preds[i].shape != tar_hsnr.shape:
                print(
                    f'Baseline prediction {i} shape {baseline_preds[i].shape} does not match target shape {tar_hsnr.shape}'
                )
                print('This happens when we want to predict the edges of the image.')
                return
    color_ch_list = ['goldenrod', 'cyan']
    color_pred = 'red'
    insetplot_xmax_value = 10000
    insetplot_xmin_value = -1000
    inset_min_labelsize = 10
    inset_rect = [0.05, 0.05, 0.4, 0.2]

    img_sz = 3
    ncols = num_samples + len(baseline_preds) + 1 + 1 + 1 + 1 + 1 * (num_samples > 1)
    grid_factor = 5
    grid_img_sz = img_sz * grid_factor
    example_spacing = 1
    c0_extra = 1
    nimgs = 1
    fig_w = ncols * img_sz + 2 * c0_extra / grid_factor
    fig_h = int(img_sz * ncols + (example_spacing * (nimgs - 1)) / grid_factor)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(nrows=int(grid_factor * fig_h), ncols=int(grid_factor * fig_w), hspace=0.2, wspace=0.2)
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    # plot baselines
    for i in range(2, 2 + len(baseline_preds)):
        for col_idx in range(baseline_preds[0].shape[0]):
            ax_temp = fig.add_subplot(gs[col_idx * grid_img_sz:grid_img_sz * (col_idx + 1),
                                         i * grid_img_sz + c0_extra:(i + 1) * grid_img_sz + c0_extra])
            print(tar_hsnr.shape, baseline_preds[i - 2].shape)
            psnr = get_psnr_str(tar_hsnr, baseline_preds[i - 2], col_idx)
            ax_temp.imshow(baseline_preds[i - 2][col_idx], cmap='magma')
            add_psnr_str(ax_temp, psnr)
            clean_ax(ax_temp)

    # plot samples
    sample_start_idx = 2 + len(baseline_preds)
    for i in range(sample_start_idx, ncols - 3):
        for col_idx in range(recon_img_list.shape[1]):
            ax_temp = fig.add_subplot(gs[col_idx * grid_img_sz:grid_img_sz * (col_idx + 1),
                                         i * grid_img_sz + c0_extra:(i + 1) * grid_img_sz + c0_extra])
            psnr = get_psnr_str(tar_hsnr, recon_img_list[i - sample_start_idx], col_idx)
            ax_temp.imshow(recon_img_list[i - sample_start_idx][col_idx], cmap='magma')
            add_psnr_str(ax_temp, psnr)
            clean_ax(ax_temp)
            # inset_ax = add_pixel_kde(ax_temp,
            #                       inset_rect,
            #                       [tar_hsnr[col_idx],
            #                        recon_img_list[i - sample_start_idx][col_idx]],
            #                       inset_min_labelsize,
            #                       label_list=['', ''],
            #                       color_list=[color_ch_list[col_idx], color_pred],
            #                       plot_xmax_value=insetplot_xmax_value,
            #                       plot_xmin_value=insetplot_xmin_value)

            # inset_ax.set_xticks([])
            # inset_ax.set_yticks([])

    # difference image
    if num_samples > 1:
        for col_idx in range(recon_img_list.shape[1]):
            ax_temp = fig.add_subplot(gs[col_idx * grid_img_sz:grid_img_sz * (col_idx + 1),
                                         (ncols - 3) * grid_img_sz + c0_extra:(ncols - 2) * grid_img_sz + c0_extra])
            ax_temp.imshow(recon_img_list[1][col_idx] - recon_img_list[0][col_idx], cmap='coolwarm')
            clean_ax(ax_temp)

    for col_idx in range(recon_img_list.shape[1]):
        # print(recon_img_list.shape)
        ax_temp = fig.add_subplot(gs[col_idx * grid_img_sz:grid_img_sz * (col_idx + 1),
                                     c0_extra + (ncols - 2) * grid_img_sz:(ncols - 1) * grid_img_sz + c0_extra])
        psnr = get_psnr_str(tar_hsnr, recon_img_list.mean(axis=0), col_idx)
        ax_temp.imshow(recon_img_list.mean(axis=0)[col_idx], cmap='magma')
        add_psnr_str(ax_temp, psnr)
        # inset_ax = add_pixel_kde(ax_temp,
        #                           inset_rect,
        #                           [tar_hsnr[col_idx],
        #                            recon_img_list.mean(axis=0)[col_idx]],
        #                           inset_min_labelsize,
        #                           label_list=['', ''],
        #                           color_list=[color_ch_list[col_idx], color_pred],
        #                           plot_xmax_value=insetplot_xmax_value,
        #                           plot_xmin_value=insetplot_xmin_value)
        # inset_ax.set_xticks([])
        # inset_ax.set_yticks([])

        clean_ax(ax_temp)

        ax_temp = fig.add_subplot(gs[col_idx * grid_img_sz:grid_img_sz * (col_idx + 1),
                                     (ncols - 1) * grid_img_sz + 2 * c0_extra:(ncols) * grid_img_sz + 2 * c0_extra])
        ax_temp.imshow(tar_hsnr[col_idx], cmap='magma')
        if col_idx == 0:
            legend_ch1_ax = ax_temp
        if col_idx == 1:
            legend_ch2_ax = ax_temp

        # inset_ax = add_pixel_kde(ax_temp,
        #                           inset_rect,
        #                           [tar_hsnr[col_idx],
        #                            ],
        #                           inset_min_labelsize,
        #                           label_list=[''],
        #                           color_list=[color_ch_list[col_idx]],
        #                           plot_xmax_value=insetplot_xmax_value,
        #                           plot_xmin_value=insetplot_xmin_value)
        # inset_ax.set_xticks([])
        # inset_ax.set_yticks([])

        clean_ax(ax_temp)

        ax_temp = fig.add_subplot(gs[col_idx * grid_img_sz:grid_img_sz * (col_idx + 1), grid_img_sz:2 * grid_img_sz])
        ax_temp.imshow(tar[0, col_idx].cpu().numpy(), cmap='magma')
        # inset_ax = add_pixel_kde(ax_temp,
        #                           inset_rect,
        #                           [tar[0,col_idx].cpu().numpy(),
        #                            ],
        #                           inset_min_labelsize,
        #                           label_list=[''],
        #                           color_list=[color_ch_list[col_idx]],
        #                           plot_kwargs_list=[{'linestyle':'--'}],
        #                           plot_xmax_value=insetplot_xmax_value,
        #                           plot_xmin_value=insetplot_xmin_value)

        # inset_ax.set_xticks([])
        # inset_ax.set_yticks([])

        clean_ax(ax_temp)

    ax_temp = fig.add_subplot(gs[0:grid_img_sz, 0:grid_img_sz])
    ax_temp.imshow(inp[0, 0].cpu().numpy(), cmap='magma')
    clean_ax(ax_temp)
    import matplotlib.lines as mlines

    # line_ch1 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[0], linestyle='-', label='$C_1$')
    # line_ch2 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[1], linestyle='-', label='$C_2$')
    # line_pred = mlines.Line2D([0, 1], [0, 1], color=color_pred, linestyle='-', label='Pred')
    # line_noisych1 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[0], linestyle='--', label='$C^N_1$')
    # line_noisych2 = mlines.Line2D([0, 1], [0, 1], color=color_ch_list[1], linestyle='--', label='$C^N_2$')
    # legend_ch1 = legend_ch1_ax.legend(handles=[line_ch1, line_noisych1, line_pred], loc='upper right', frameon=False, labelcolor='white',
    #                         prop={'size': 11})
    # legend_ch2 = legend_ch2_ax.legend(handles=[line_ch2, line_noisych2, line_pred], loc='upper right', frameon=False, labelcolor='white',
    #                             prop={'size': 11})

    if calibration_stats is not None:
        smaller_offset = 4
        ax_temp = fig.add_subplot(gs[grid_img_sz + 1:2 * grid_img_sz - smaller_offset + 1,
                                     smaller_offset - 1:grid_img_sz - 1])
        plot_calibration(ax_temp, calibration_stats)
