from denoisplit.nets.lvae import LadderVAE, compute_batch_mean, torch_nanmean
import torch.nn as nn
import torch.optim as optim
from denoisplit.core.likelihoods import GaussianLikelihoodWithStitching
import torch
import torchvision.transforms.functional as F
from denoisplit.core.psnr import RangeInvariantPsnr
import numpy as np


class SqueezeLayer(nn.Module):
    def forward(self, x):
        return torch.squeeze(x)


class LadderVAEwithStitching(LadderVAE):
    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at=use_uncond_mode_at, target_ch=target_ch)
        self.offset_prediction_input_z_idx = config.model.offset_prediction_input_z_idx
        latent_spatial_dims = config.data.image_size
        if config.model.decoder.multiscale_retain_spatial_dims is False or config.data.multiscale_lowres_count is None:
            latent_spatial_dims = latent_spatial_dims // np.power(2, 1 + self.offset_prediction_input_z_idx)
        in_channels = config.model.z_dims[self.offset_prediction_input_z_idx]
        offset_latent_dims = config.model.offset_latent_dims
        self.nbr_set_count = config.data.get('nbr_set_count', None)
        self.regularize_offset = config.model.get('regularize_offset', False)
        self._offset_reg_w = None
        if self.regularize_offset:
            self._offset_reg_w = config.model.offset_regularization_w

        if config.model.get('offset_prediction_scalar_prediction', False):
            output_ch = 1
        else:
            output_ch = 2

        self.offset_predictor = nn.Sequential(
            nn.Conv2d(in_channels, offset_latent_dims, 1),
            self.get_nonlin()(),
            nn.AvgPool2d(latent_spatial_dims),
            SqueezeLayer(),
            nn.Linear(offset_latent_dims, output_ch,
                      bias=output_ch != 1),  # If we predict just one value, then bias is not needed
        )

    def create_likelihood_module(self):
        self.likelihood = GaussianLikelihoodWithStitching(self.decoder_n_filters,
                                                          self.target_ch,
                                                          predict_logvar=self.predict_logvar,
                                                          logvar_lowerbound=self.logvar_lowerbound)

    def lowres_inputbranch_parameters(self):
        if self.lowres_first_bottom_ups is not None:
            return list(self.lowres_first_bottom_ups.parameters())
        return []

    def configure_optimizers(self):
        params1 = list(self.first_bottom_up.parameters()) + list(self.bottom_up_layers.parameters()) + list(
            self.top_down_layers.parameters()) + list(self.final_top_down.parameters()) + list(
                self.likelihood.parameters()) + self.lowres_inputbranch_parameters()

        optimizer1 = optim.Adamax(params1, lr=self.lr, weight_decay=0)
        params2 = self.offset_predictor.parameters()
        optimizer2 = optim.Adamax(params2, lr=self.lr, weight_decay=0)

        scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1,
                                                          self.lr_scheduler_mode,
                                                          patience=self.lr_scheduler_patience,
                                                          factor=0.5,
                                                          min_lr=1e-12,
                                                          verbose=True)

        scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2,
                                                          self.lr_scheduler_mode,
                                                          patience=self.lr_scheduler_patience,
                                                          factor=0.5,
                                                          min_lr=1e-12,
                                                          verbose=True)

        return [optimizer1, optimizer2], [{
            'scheduler': scheduler1,
            'monitor': self.lr_scheduler_monitor
        }, {
            'scheduler': scheduler2,
            'monitor': self.lr_scheduler_monitor
        }]

    def _get_reconstruction_loss_vector(self, reconstruction, input, offset, return_predicted_img=False):
        """
        Args:
            return_predicted_img: If set to True, the besides the loss, the reconstructed image is also returned.
        """

        # Log likelihood
        ll, like_dict = self.likelihood(reconstruction, input, offset)

        recons_loss = compute_batch_mean(-1 * ll)
        output = {
            'loss': recons_loss,
            'ch1_loss': compute_batch_mean(-ll[:, 0]),
            'ch2_loss': compute_batch_mean(-ll[:, 1]),
        }

        if return_predicted_img:
            return output, like_dict['params']['mean']

        return output

    def get_reconstruction_loss(self, reconstruction, input, offset, return_predicted_img=False):
        output = self._get_reconstruction_loss_vector(reconstruction,
                                                      input,
                                                      offset,
                                                      return_predicted_img=return_predicted_img)
        loss_dict = output[0] if return_predicted_img else output
        loss_dict['loss'] = torch.mean(loss_dict['loss'])
        loss_dict['ch1_loss'] = torch.mean(loss_dict['ch1_loss'])
        loss_dict['ch2_loss'] = torch.mean(loss_dict['ch2_loss'])

        if return_predicted_img:
            assert len(output) == 2
            return loss_dict, output[1]
        else:
            return loss_dict

    def compute_offset(self, z_arr):
        offset = self.offset_predictor(z_arr[self.offset_prediction_input_z_idx])
        # In case of a scalar prediction
        if offset.shape[-1] == 1:
            offset = torch.cat([offset, -1 * offset], dim=-1)

        return offset[..., None, None]

    def training_step(self, batch: tuple, batch_idx: int, optimizer_idx: int, enable_logging=True):
        x, target, grid_sizes = batch

        if optimizer_idx == 0 and self.nbr_set_count is not None:
            mask = np.arange(len(x)) >= 5 * self.nbr_set_count
            x = x[mask]
            target = target[mask]
            grid_sizes = grid_sizes[mask]

        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)

        out, td_data = self.forward(x_normalized)
        offset = self.compute_offset(td_data['z'])
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict, imgs = self.get_reconstruction_loss(out, target_normalized, offset, return_predicted_img=True)

        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        recons_loss = recons_loss_dict['loss']

        kl_loss = self.get_kl_divergence_loss(td_data)
        if optimizer_idx == 0:
            net_loss = recons_loss + self.get_kl_weight() * kl_loss
            if enable_logging:
                for i, x in enumerate(td_data['debug_qvar_max']):
                    self.log(f'qvar_max:{i}', x.item(), on_epoch=True)

                self.log('reconstruction_loss', recons_loss_dict['loss'], on_epoch=True)
                self.log('kl_loss', kl_loss, on_epoch=True)
                self.log('training_loss', net_loss, on_epoch=True)
                self.log('lr', self.lr, on_epoch=True)
                self.log('grad_norm_bottom_up', self.grad_norm_bottom_up, on_epoch=True)
                self.log('grad_norm_top_down', self.grad_norm_top_down, on_epoch=True)

        elif optimizer_idx == 1:
            nbr_cons_loss = self.nbr_consistency_loss.get(imgs, grid_sizes=grid_sizes)
            offset_reg_loss = 0.0
            if self.regularize_offset:
                offset_reg_loss = torch.norm(offset)
                offset_reg_loss = self._offset_reg_w * offset_reg_loss
                self.log('offset_reg_loss', offset_reg_loss.item(), on_epoch=True)

            if nbr_cons_loss is not None:
                nbr_cons_loss = self.nbr_consistency_w * nbr_cons_loss
                self.log('nbr_cons_loss', nbr_cons_loss.item(), on_epoch=True)
                net_loss = nbr_cons_loss + offset_reg_loss

        output = {
            'loss': net_loss,
        }
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if net_loss is None or torch.isnan(net_loss).any():
            return None

        return output

    def validation_step(self, batch, batch_idx):
        x, target = batch[:2]
        self.set_params_to_same_device_as(target)

        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)
        out, td_data = self.forward(x_normalized)
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        offset = self.compute_offset(td_data['z'])

        recons_loss_dict, recons_img = self.get_reconstruction_loss(out,
                                                                    target_normalized,
                                                                    offset,
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
                sample, _ = self(x_normalized[0:1, ...])
                sample = self.likelihood.get_mean_lv(sample)[0]
                all_samples.append(sample[None])

            all_samples = torch.cat(all_samples, dim=0)
            all_samples = all_samples * self.data_std + self.data_mean
            all_samples = all_samples.cpu()
            img_mmse = torch.mean(all_samples, dim=0)[0]
            self.log_images_for_tensorboard(all_samples[:, 0, 0, ...], target[0, 0, ...], img_mmse[0], 'label1')
            self.log_images_for_tensorboard(all_samples[:, 0, 1, ...], target[0, 1, ...], img_mmse[1], 'label2')


if __name__ == '__main__':
    from denoisplit.configs.lvae_with_stitch_config import get_config
    import torch
    config = get_config()
    model = LadderVAEwithStitching(0, 1, config)
    inp = torch.rand((16, 1, 64, 64))
    tar = torch.rand((16, 2, 64, 64))
    grid_sizes = torch.Tensor([32] * 5 + [40] * 5 + [24] * 5 + [41]).type(torch.int32)

    model.validation_step((inp, tar, grid_sizes), 0)
    loss0 = model.training_step((inp, tar, grid_sizes), 0, 0)
    loss1 = model.training_step((inp, tar, grid_sizes), 0, 1)
