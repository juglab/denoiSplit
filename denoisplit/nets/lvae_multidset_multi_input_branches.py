from typing import List

import torch

from denoisplit.core.data_utils import crop_img_tensor
from denoisplit.core.loss_type import LossType
from denoisplit.core.psnr import RangeInvariantPsnr
from denoisplit.nets.lvae import torch_nanmean
from denoisplit.nets.lvae_twodset import LadderVaeTwoDset


class LadderVaeMultiDatasetMultiBranch(LadderVaeTwoDset):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at, target_ch)
        stride = 1 if config.model.no_initial_downscaling else 2
        del self.first_bottom_up
        self._first_bottom_up_subdset0 = self.create_first_bottom_up(stride)
        self._first_bottom_up_subdset1 = self.create_first_bottom_up(stride)

    def forward(self, x, loss_idx: int):
        img_size = x.size()[2:]

        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad, loss_idx)
        mode_layers = range(self.n_layers) if self.non_stochastic_version else None
        # Top-down inference/generation
        out, td_data = self.topdown_pass(bu_values, mode_layers=mode_layers)

        if out.shape[-1] > img_size[-1]:
            # Restore original image size
            out = crop_img_tensor(out, img_size)

        return out, td_data

    def bottomup_pass(self, inp, loss_idx):
        if loss_idx == LossType.ElboMixedReconstruction:
            return self._bottomup_pass(inp, self._first_bottom_up_subdset0, self.lowres_first_bottom_ups,
                                       self.bottom_up_layers)

        elif loss_idx == LossType.Elbo:
            return self._bottomup_pass(inp, self._first_bottom_up_subdset1, self.lowres_first_bottom_ups,
                                       self.bottom_up_layers)

    def merge_td_data(self, td_data1, len1: int, td_data2, len2: int):
        """
        merge the td data
        """
        if td_data1 is None:
            return td_data2
        if td_data2 is None:
            return td_data1

        output_td_data = {}
        for key in ['z', 'kl']:
            output_td_data[key] = []
            for i in range(len(td_data1[key])):
                concat_value = torch.cat([td_data1[key][i], td_data2[key][i]], dim=0)
                output_td_data[key].append(concat_value)

        for key in ['debug_qvar_max']:
            output_td_data[key] = []
            for i in range(len(td_data1[key])):
                merged_value = torch.max(td_data1[key][i], td_data2[key][i])
                output_td_data[key].append(merged_value)

        return output_td_data

    def merge_vectors(self, vector_tuple1: List[torch.Tensor], vector_tuple2: List[torch.Tensor]):
        out_vectors = []
        for i in range(len(vector_tuple1)):
            if vector_tuple1[i] is None or torch.numel(vector_tuple1[i]) == 0:
                out_vectors.append(vector_tuple2[i])
            elif vector_tuple2[i] is None or torch.numel(vector_tuple2[i]) == 0:
                out_vectors.append(vector_tuple1[i])
            else:
                out_vectors.append(torch.cat([vector_tuple1[i], vector_tuple2[i]], dim=0))
        return out_vectors

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target, dset_idx, loss_idx = batch
        assert self.normalized_input == True
        x_normalized = x
        target_normalized = self.normalize_target(target, dset_idx)

        mask_mixrecons = loss_idx == LossType.ElboMixedReconstruction
        mask_2ch = loss_idx == LossType.Elbo
        assert torch.sum(mask_2ch) + torch.sum(mask_mixrecons) == len(target)
        if mask_mixrecons.sum() > 0:
            out_mixrecons, td_data_mixrecons = self.forward(x_normalized[mask_mixrecons],
                                                            LossType.ElboMixedReconstruction)
        else:
            out_mixrecons = None
            td_data_mixrecons = None

        if mask_2ch.sum() > 0:
            out_2ch, td_data_2ch = self.forward(x_normalized[mask_2ch], LossType.Elbo)
        else:
            out_2ch = None
            td_data_2ch = None

        td_data = self.merge_td_data(td_data_mixrecons, mask_mixrecons.sum(), td_data_2ch, mask_2ch.sum())

        assert self.encoder_no_padding_mode is False

        out, target_normalized, dset_idx, loss_idx = self.merge_vectors(
            (out_mixrecons, target_normalized[mask_mixrecons], dset_idx[mask_mixrecons], loss_idx[mask_mixrecons]),
            (out_2ch, target_normalized[mask_2ch], dset_idx[mask_2ch], loss_idx[mask_2ch]),
        )

        recons_loss_dict = self.get_reconstruction_loss(out,
                                                        target_normalized,
                                                        dset_idx,
                                                        loss_idx,
                                                        return_predicted_img=False)

        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        recons_loss = recons_loss_dict['loss']
        if self.loss_type == LossType.ElboMixedReconstruction:
            recons_loss += self.mixed_rec_w * recons_loss_dict['mixed_loss']

            if enable_logging:
                self.log('mixed_reconstruction_loss', recons_loss_dict['mixed_loss'], on_epoch=True)

        if self.non_stochastic_version:
            kl_loss = torch.Tensor([0.0]).cuda()
            net_loss = recons_loss
        else:
            kl_loss = self.get_kl_divergence_loss(td_data)
            net_loss = recons_loss + self.get_kl_weight() * kl_loss

        if enable_logging:
            for i, x in enumerate(td_data['debug_qvar_max']):
                self.log(f'qvar_max:{i}', x.item(), on_epoch=True)

            self.log('reconstruction_loss', recons_loss_dict['loss'], on_epoch=True)
            self.log('kl_loss', kl_loss, on_epoch=True)
            self.log('training_loss', net_loss, on_epoch=True)
            self.log('lr', self.lr, on_epoch=True)
            if self._interchannel_weights is not None:
                self.log('interchannel_w0', self._interchannel_weights.squeeze()[0].item(), on_epoch=True)
                self.log('interchannel_w1', self._interchannel_weights.squeeze()[1].item(), on_epoch=True)

            # self.log('grad_norm_bottom_up', self.grad_norm_bottom_up, on_epoch=True)
            # self.log('grad_norm_top_down', self.grad_norm_top_down, on_epoch=True)

        output = {
            'loss': net_loss,
            'reconstruction_loss': recons_loss,
            'kl_loss': self.get_kl_weight() * kl_loss,
        }

        if self.loss_type == LossType.ElboMixedReconstruction:
            output['mixed_loss'] = self.mixed_rec_w * recons_loss_dict['mixed_loss']

        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output

    def validation_step(self, batch, batch_idx):
        x, target, dset_idx, loss_idx = batch
        self.set_params_to_same_device_as(target)

        x_normalized = x
        target_normalized = self.normalize_target(target, dset_idx)

        mask_mixrecons = loss_idx == LossType.ElboMixedReconstruction
        mask_2ch = loss_idx == LossType.Elbo
        assert mask_2ch.sum() == len(x)
        assert mask_mixrecons.sum() == 0
        out, td_data = self.forward(x_normalized, LossType.Elbo)

        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict, recons_img = self.get_reconstruction_loss(out,
                                                                    target_normalized,
                                                                    dset_idx,
                                                                    loss_idx,
                                                                    return_predicted_img=True)
        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        self.label1_psnr.update(recons_img[:, 0], target_normalized[:, 0])
        self.label2_psnr.update(recons_img[:, 1], target_normalized[:, 1])

        psnr_label1 = RangeInvariantPsnr(target_normalized[:, 0].clone(), recons_img[:, 0].clone())
        psnr_label2 = RangeInvariantPsnr(target_normalized[:, 1].clone(), recons_img[:, 1].clone())
        recons_loss = recons_loss_dict['loss']
        # kl_loss = self.get_kl_divergence_loss(td_data)
        # net_loss = recons_loss + self.get_kl_weight() * kl_loss
        self.log('val_loss', recons_loss, on_epoch=True)
        val_psnr_l1 = torch_nanmean(psnr_label1).item()
        val_psnr_l2 = torch_nanmean(psnr_label2).item()
        self.log('val_psnr_l1', val_psnr_l1, on_epoch=True)
        self.log('val_psnr_l2', val_psnr_l2, on_epoch=True)
        # self.log('val_psnr', (val_psnr_l1 + val_psnr_l2) / 2, on_epoch=True)

        if batch_idx == 0 and self.power_of_2(self.current_epoch):
            all_samples = []
            for i in range(20):
                sample, _ = self(x_normalized[0:1, ...], LossType.Elbo)
                sample = self.likelihood.get_mean_lv(sample)[0]
                all_samples.append(sample[None])

            all_samples = torch.cat(all_samples, dim=0)
            data_mean, data_std = self.get_mean_std_for_one_batch(dset_idx, self.data_mean, self.data_std)
            all_samples = all_samples * data_std['target'] + data_mean['target']
            all_samples = all_samples.cpu()
            img_mmse = torch.mean(all_samples, dim=0)[0]
            self.log_images_for_tensorboard(all_samples[:, 0, 0, ...], target[0, 0, ...], img_mmse[0], 'label1')
            self.log_images_for_tensorboard(all_samples[:, 0, 1, ...], target[0, 1, ...], img_mmse[1], 'label2')


if __name__ == '__main__':
    from denoisplit.configs.ht_iba1_ki64_multidata_config import get_config

    data_mean = {
        'subdset_0': {
            'target': torch.Tensor([1.1, 3.2]).reshape((1, 2, 1, 1)),
            'input': torch.Tensor([1366]).reshape((1, 1, 1, 1))
        },
        'subdset_1': {
            'target': torch.Tensor([15, 30]).reshape((1, 2, 1, 1)),
            'input': torch.Tensor([10]).reshape((1, 1, 1, 1))
        }
    }

    data_std = {
        'subdset_0': {
            'target': torch.Tensor([21, 45]).reshape((1, 2, 1, 1)),
            'input': torch.Tensor([955]).reshape((1, 1, 1, 1))
        },
        'subdset_1': {
            'target': torch.Tensor([90, 2]).reshape((1, 2, 1, 1)),
            'input': torch.Tensor([121]).reshape((1, 1, 1, 1))
        }
    }

    config = get_config()
    model = LadderVaeMultiDatasetMultiBranch(data_mean, data_std, config)

    dset_idx = torch.Tensor([0, 1, 0, 1])
    loss_idx = torch.Tensor(
        [LossType.Elbo, LossType.ElboMixedReconstruction, LossType.Elbo, LossType.ElboMixedReconstruction])
    x = torch.rand((4, 1, 64, 64))
    target = torch.rand((4, 2, 64, 64))
    batch = (x, target, dset_idx, loss_idx)
    model.training_step(batch, 0, enable_logging=True)
    model.validation_step(batch, 0)
