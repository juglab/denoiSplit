from denoisplit.nets.lvae import LadderVAE
import torch.nn as nn
import torch
from denoisplit.core.loss_type import LossType


class LadderVAEMultiTarget(LadderVAE):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super(LadderVAEMultiTarget, self).__init__(data_mean,
                                                   data_std,
                                                   config,
                                                   use_uncond_mode_at=use_uncond_mode_at,
                                                   target_ch=target_ch)
        self._lres_final_top_down = None
        self._latent_dims = config.model.z_dims
        self._lres_final_top_down = nn.ModuleList()
        self._lres_conv_for_z = nn.ModuleList()

        for ith_res in range(self._multiscale_count - 1):
            self._lres_conv_for_z.append(
                nn.Conv2d(self._latent_dims[ith_res], config.model.decoder.n_filters, 3, padding=1))
            self._lres_final_top_down.append(self.create_final_topdown_layer(False))

        self._lres_likelihoods = None
        self._lres_likelihoods = nn.ModuleList()
        for _ in range(self._multiscale_count - 1):
            self._lres_likelihoods.append(self.create_likelihood_module())
        self._lres_recloss_w = config.loss.lres_recloss_w
        assert len(self._lres_recloss_w) == config.data.multiscale_lowres_count

        print(f'[{self.__class__.__name__}] LowResSupLen:{len(self._lres_likelihoods)} rec_w:{self._lres_recloss_w}')

    def validation_step(self, batch, batch_idx):
        x, target = batch
        return super().validation_step((x, target[:, 0]), batch_idx)

    def get_allres_predictions(self, x_normalized):
        """
        Get all disentangled predictions at all levels.
        Args:
            x_normalized:

        Returns:

        """
        out, td_data = self.forward(x_normalized)
        lowres_outs = [self.likelihood.parameter_net(out)]
        for l_to_h_idx in range(self._multiscale_count - 1):
            out_temp = self._lres_conv_for_z[l_to_h_idx](td_data['z'][l_to_h_idx])
            lowres_out = self._lres_final_top_down[l_to_h_idx](out_temp)
            lowres_out = self._lres_likelihoods[l_to_h_idx].parameter_net(lowres_out)
            lowres_outs.append(lowres_out)
        return lowres_outs

    def get_all_res_reconstruction_loss(self, out, td_data, target_normalized):
        """
        Reconstruction loss from all resolutions
        """
        lowres_outs = []
        for l_to_h_idx in range(self._multiscale_count - 1):
            out_temp = self._lres_conv_for_z[l_to_h_idx](td_data['z'][l_to_h_idx])
            lowres_outs.append(self._lres_final_top_down[l_to_h_idx](out_temp))

        recons_loss = 0
        assert self._multiscale_count == target_normalized.shape[1]

        for ith_res in range(self._multiscale_count):
            if ith_res == 0:
                recons_loss_dict = self.get_reconstruction_loss(out, target_normalized[:, 0])
            else:
                new_sz = self.img_shape[0] // (2**ith_res)
                skip_idx = (target_normalized.shape[-1] - new_sz) // 2
                tar_res = target_normalized[:, ith_res, :, skip_idx:-skip_idx, skip_idx:-skip_idx]
                lowres_pred = lowres_outs[ith_res - 1]
                if self.multiscale_decoder_retain_spatial_dims:
                    lowres_pred = lowres_pred[:, :, skip_idx:-skip_idx, skip_idx:-skip_idx]

                recons_loss_dict = self.get_reconstruction_loss(lowres_pred,
                                                                tar_res,
                                                                likelihood_obj=self._lres_likelihoods[ith_res - 1])
            recons_loss += recons_loss_dict['loss'] * self._lres_recloss_w[ith_res]
        return recons_loss

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target = batch
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)

        out, td_data = self.forward(x_normalized)
        recons_loss = self.get_all_res_reconstruction_loss(out, td_data, target_normalized)
        kl_loss = self.get_kl_divergence_loss(td_data)
        net_loss = recons_loss + self.get_kl_weight() * kl_loss
        assert self.loss_type not in [LossType.ElboMixedReconstruction, LossType.ElboWithNbrConsistency]
        assert self.non_stochastic_version is False

        if enable_logging:
            for i, x in enumerate(td_data['debug_qvar_max']):
                self.log(f'qvar_max:{i}', x.item(), on_epoch=True)

            self.log('reconstruction_loss', recons_loss, on_epoch=True)
            self.log('kl_loss', kl_loss, on_epoch=True)
            self.log('training_loss', net_loss, on_epoch=True)
            self.log('lr', self.lr, on_epoch=True)
            self.log('grad_norm_bottom_up', self.grad_norm_bottom_up, on_epoch=True)
            self.log('grad_norm_top_down', self.grad_norm_top_down, on_epoch=True)

        output = {
            'loss': net_loss,
            'reconstruction_loss': recons_loss.detach(),
            'kl_loss': kl_loss.detach(),
        }
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output
