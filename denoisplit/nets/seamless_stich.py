"""
Do seamless stitching
"""
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.optim as optim
from denoisplit.core.seamless_stitch_base import SeamlessStitchBase


class Model(nn.Module):
    def __init__(self, num_samples, N):
        super().__init__()
        self._N = N
        self.params = nn.Parameter(torch.zeros(num_samples, self._N, self._N))
        self.shape = self.params.shape

    def __getitem__(self, pos):
        i, j = pos
        return self.params[:, i, j]


class SeamlessStitch(SeamlessStitchBase):
    def __init__(self, grid_size, stitched_frame, learning_rate, lr_patience=10, lr_reduction_factor=0.1):
        super().__init__(grid_size, stitched_frame)
        self.params = Model(len(stitched_frame), self._N)
        self.opt = torch.optim.SGD(self.params.parameters(), lr=learning_rate)
        self.loss_metric = nn.L1Loss(reduction='sum')

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt,
                                                                 'min',
                                                                 patience=lr_patience,
                                                                 factor=lr_reduction_factor,
                                                                 threshold_mode='abs',
                                                                 min_lr=1e-12,
                                                                 verbose=True)
        print(
            f'[{self.__class__.__name__}] Grid:{grid_size} LR:{learning_rate} LP:{lr_patience} LRF:{lr_reduction_factor}'
        )

    def get_ch0_offset(self, row_idx, col_idx):
        return self.params[row_idx, col_idx].detach().cpu().numpy()[:, None, None]

    def _compute_loss_on_boundaries(self, boundary1, boundary2, boundary1_offset):
        # return torch.Tensor([0.0])
        ch0_loss = self.loss_metric(boundary1[:, 0] + boundary1_offset[..., None], boundary2[:, 0])
        ch1_loss = self.loss_metric(boundary1[:, 1] - boundary1_offset[..., None], boundary2[:, 1])

        return (ch0_loss + ch1_loss) / 2

    def _compute_left_loss(self, row_idx, col_idx):
        if col_idx == 0:
            return 0.0
        p = self.params[row_idx, col_idx]

        left_p_boundary = self.get_lboundary(row_idx, col_idx)
        right_p_boundary = self.get_rboundary(row_idx, col_idx - 1)
        return (left_p_boundary, right_p_boundary, p)

    def _compute_right_loss(self, row_idx, col_idx):
        if col_idx == self.params.shape[1] - 1:
            return 0.0
        p = self.params[row_idx, col_idx]

        left_p_boundary = self.get_lboundary(row_idx, col_idx + 1)
        right_p_boundary = self.get_rboundary(row_idx, col_idx)
        return (right_p_boundary, left_p_boundary, p)

    def _compute_top_loss(self, row_idx, col_idx):
        if row_idx == 0:
            return 0.0
        p = self.params[row_idx, col_idx]

        top_p_boundary = self.get_tboundary(row_idx, col_idx)
        bottom_p_boundary = self.get_bboundary(row_idx - 1, col_idx)
        return (top_p_boundary, bottom_p_boundary, p)

    def _compute_bottom_loss(self, row_idx, col_idx):
        if row_idx == self.params.shape[1] - 1:
            return 0.0
        p = self.params[row_idx, col_idx]

        top_p_boundary = self.get_tboundary(row_idx + 1, col_idx)
        bottom_p_boundary = self.get_bboundary(row_idx, col_idx)
        return (bottom_p_boundary, top_p_boundary, p)

    def _compute_loss(self,
                      row_idx,
                      col_idx,
                      compute_left=True,
                      compute_right=True,
                      compute_top=True,
                      compute_bottom=True):
        left_loss = self._compute_left_loss(row_idx, col_idx) if compute_left else None
        right_loss = self._compute_right_loss(row_idx, col_idx) if compute_right else None

        top_loss = self._compute_top_loss(row_idx, col_idx) if compute_top else None
        bottom_loss = self._compute_bottom_loss(row_idx, col_idx) if compute_bottom else None

        b1_arr = []
        b2_arr = []
        offset_arr = []
        if left_loss is not None:
            b1_arr.append(left_loss[0])
            b2_arr.append(left_loss[1])
            offset_arr.append(left_loss[2])

        if right_loss is not None:
            b1_arr.append(right_loss[0])
            b2_arr.append(right_loss[1])
            offset_arr.append(right_loss[2])

        if top_loss is not None:
            b1_arr.append(top_loss[0])
            b2_arr.append(top_loss[1])
            offset_arr.append(top_loss[2])

        if bottom_loss is not None:
            b1_arr.append(bottom_loss[0])
            b2_arr.append(bottom_loss[1])
            offset_arr.append(bottom_loss[2])

        return b1_arr, b2_arr, offset_arr

    def compute_loss(self,
                     batch_size=100,
                     compute_left=True,
                     compute_right=True,
                     compute_top=True,
                     compute_bottom=True):
        loss = 0.0
        b1_arr = []
        b2_arr = []
        offset_arr = []
        loss = 0.0

        normalizing_factor = self._data.shape[0] * (2 * ((self._N - 1)**2))
        for row_idx in range(self._N):
            for col_idx in range(self._N):
                a, b, c = self._compute_loss(row_idx,
                                             col_idx,
                                             compute_left=compute_left,
                                             compute_right=compute_right,
                                             compute_top=compute_top,
                                             compute_bottom=compute_bottom)
                b1_arr += a
                b2_arr += b
                offset_arr += c
                if batch_size <= len(b1_arr):
                    loss += self._compute_loss_on_boundaries(torch.cat(b1_arr, dim=0), torch.cat(b2_arr, dim=0),
                                                             torch.cat(offset_arr, dim=0)) / normalizing_factor
                    b1_arr = []
                    b2_arr = []
                    offset_arr = []

        if len(offset_arr):
            loss += self._compute_loss_on_boundaries(torch.cat(b1_arr, dim=0), torch.cat(b2_arr, dim=0),
                                                     torch.cat(offset_arr, dim=0)) / normalizing_factor
        return loss

    def fit(self, batch_size=512, steps=100):
        loss_arr = []
        steps_iter = tqdm(range(steps))
        for _ in steps_iter:
            self.params.zero_grad()
            loss = self.compute_loss(batch_size=batch_size)
            loss.backward()
            self.opt.step()

            loss_arr.append(loss.item())
            steps_iter.set_description(f'Loss: {loss_arr[-1]:.3f}')
            self.lr_scheduler.step(loss)

        return loss_arr
