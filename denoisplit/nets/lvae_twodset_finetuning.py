from copy import deepcopy

import torch
import torch.optim as optim

import ml_collections
from denoisplit.core.likelihoods import GaussianLikelihood, NoiseModelLikelihood
from denoisplit.core.loss_type import LossType
from denoisplit.core.model_type import ModelType
from denoisplit.core.psnr import RangeInvariantPsnr
from denoisplit.loss.restricted_reconstruction_loss import RestrictedReconstruction
from denoisplit.nets.lvae import compute_batch_mean, torch_nanmean
from denoisplit.nets.lvae_twodset_restrictedrecons import LadderVaeTwoDsetRestrictedRecons
from denoisplit.nets.noise_model import get_noise_model


class LadderVaeTwoDsetFinetuning(LadderVaeTwoDsetRestrictedRecons):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2, val_idx_manager=None):
        super(LadderVaeTwoDsetRestrictedRecons, self).__init__(data_mean,
                                                               data_std,
                                                               config,
                                                               use_uncond_mode_at=use_uncond_mode_at,
                                                               target_ch=target_ch,
                                                               val_idx_manager=val_idx_manager)
        self.rest_recons_loss = None
        self.mixed_rec_w = config.loss.mixed_rec_weight

        self.split_w = config.loss.split_weight
        self.init_normalization(data_mean, data_std)
        self.likelihood_old = self.likelihood
        new_config = ml_collections.ConfigDict()
        new_config.data = ml_collections.ConfigDict()
        for key in config.data.dset1:
            new_config.data[key] = config.data.dset1[key]

        self._interchannel_weights = None
        new_config.model = ml_collections.ConfigDict()
        new_config.model.enable_noise_model = True
        new_config.model.noise_model_ch1_fpath = config.model.finetuning_noise_model_ch1_fpath
        new_config.model.noise_model_ch2_fpath = config.model.finetuning_noise_model_ch2_fpath
        new_config.model.noise_model_type = config.model.finetuning_noise_model_type
        new_config.model.model_type = ModelType.Denoiser
        new_config.model.denoise_channel = 'input'
        self.noiseModel_finetuning = get_noise_model(new_config)
        mean_dict = deepcopy(self.data_mean['subdset_1'])
        std_dict = deepcopy(self.data_std['subdset_1'])
        mean_dict['target'] = mean_dict['input']
        std_dict['target'] = std_dict['input']
        self.likelihood_finetuning = NoiseModelLikelihood(self.decoder_n_filters, 1, mean_dict, std_dict,
                                                          self.noiseModel_finetuning)
        assert self.likelihood_form == 'gaussian'
        # self.likelihood = NoiseModelLikelihood(self.decoder_n_filters, self.target_ch, self.data_mean['subdset_0'],
        #                                        self.data_std['subdset_0'], self.noiseModel)
        self.likelihood = GaussianLikelihood(self.decoder_n_filters,
                                             self.target_ch,
                                             predict_logvar=self.predict_logvar,
                                             logvar_lowerbound=self.logvar_lowerbound,
                                             conv2d_bias=self.topdown_conv2d_bias)

        if config.loss.loss_type == LossType.ElboRestrictedReconstruction:
            self.rest_recons_loss = RestrictedReconstruction(1,
                                                             self.mixed_rec_w,
                                                             custom_loss_fn=self.get_loss_fn(
                                                                 self.likelihood_finetuning))
            self.rest_recons_loss.enable_nonorthogonal()

            self.automatic_optimization = False

    @staticmethod
    def get_loss_fn(likelihood_fn):

        def loss_fn(tar, pred):
            """
            Batch * H * W shape for both inputs.
            """
            mixed_recons_ll = likelihood_fn.log_likelihood(tar[:, None], {'mean': pred[:, None], 'logvar': None})
            nll = (-1 * mixed_recons_ll).mean()
            return nll

        return loss_fn

    def configure_optimizers(self):
        selected_params = []
        for name, param in self.named_parameters():
            # print(name)
            # first_bottom_up
            # final_top_down
            name = name.split('.')[0]
            if name in ['first_bottom_up', 'bottom_up_layers']:  #, 'final_top_down']:
                selected_params.append(param)

        optimizer = optim.Adamax(selected_params, lr=self.lr, weight_decay=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         self.lr_scheduler_mode,
                                                         patience=self.lr_scheduler_patience,
                                                         factor=0.5,
                                                         min_lr=1e-12,
                                                         verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.lr_scheduler_monitor}

    def _training_manual_step(self, batch, batch_idx, enable_logging=True):
        x, target, dset_idx, loss_idx = batch
        # ensure that we have exactly 16 dset 0 examples.
        csum = (dset_idx == 0).cumsum(dim=0)
        if csum[-1] < 16:
            return None
        csum_mask = csum <= 16
        # csum_mask = dset_idx == 0
        x = x[csum_mask]

        target = target[csum_mask]
        dset_idx = dset_idx[csum_mask]
        loss_idx = loss_idx[csum_mask]

        assert len(torch.unique(loss_idx[dset_idx == 0])) <= 1
        assert len(torch.unique(loss_idx[dset_idx == 1])) <= 1
        assert len(torch.unique(loss_idx)) <= 2

        optim = self.optimizers()
        optim.zero_grad()

        assert self.normalized_input == True
        x_normalized = x
        target_normalized = self.normalize_target(target, dset_idx)

        out, td_data = self.forward(x_normalized)

        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict = self.get_reconstruction_loss(out,
                                                        target_normalized,
                                                        x_normalized,
                                                        dset_idx,
                                                        loss_idx,
                                                        return_predicted_img=False)

        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        recons_loss = self.split_w * recons_loss_dict['loss']
        mask = loss_idx == LossType.Elbo
        if self.non_stochastic_version:
            kl_loss = torch.Tensor([0.0]).cuda()
            net_loss = recons_loss
        else:
            kl_dict = {'kl': [kl_level[mask] for kl_level in td_data['kl']]}
            kl_loss = self.get_kl_divergence_loss(kl_dict)
            net_loss = recons_loss + self.get_kl_weight() * kl_loss

        if isinstance(net_loss, torch.Tensor):
            self.manual_backward(net_loss, retain_graph=True)
        else:
            assert net_loss == 0.0
            return None

        if self.predict_logvar is not None:
            assert target_normalized.shape[1] * 2 == out.shape[1]
            out = out.chunk(2, dim=1)[0]

        assert target_normalized.shape[1] == out.shape[1]
        mixed_loss = None
        if (~mask).sum() > 0:
            pred_x_normalized, _ = self.get_mixed_prediction(out[~mask], None, dset_idx[~mask])
            params = list(self.named_parameters())
            relevant_params = []
            for name, param in params:
                if param.requires_grad == False:
                    pass
                else:
                    relevant_params.append((name, param))

            _ = self.rest_recons_loss.update_gradients(relevant_params, x_normalized[~mask], target_normalized[mask],
                                                       out[mask], pred_x_normalized, self.current_epoch)
        optim.step()

        if enable_logging:
            for i, x in enumerate(td_data['debug_qvar_max']):
                self.log(f'qvar_max:{i}', x.item(), on_epoch=True)

            self.log('reconstruction_loss', recons_loss_dict['loss'], on_epoch=True)
            self.log('kl_loss', kl_loss, on_epoch=True)
            self.log('training_loss', net_loss, on_epoch=True)
            self.log('lr', self.lr, on_epoch=True)
            if mixed_loss is not None:
                self.log('mixed_loss', mixed_loss)
            # self.log('grad_norm_bottom_up', self.grad_norm_bottom_up, on_epoch=True)
            # self.log('grad_norm_top_down', self.grad_norm_top_down, on_epoch=True)

        output = {
            'loss': net_loss,
            'reconstruction_loss': recons_loss.detach() if isinstance(recons_loss, torch.Tensor) else recons_loss,
            'kl_loss': kl_loss.detach(),
        }
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output

    def training_step(self, batch, batch_idx, enable_logging=True):
        if self.automatic_optimization is False:
            return self._training_manual_step(batch, batch_idx, enable_logging=enable_logging)

        x, target, dset_idx, loss_idx = batch

        assert self.normalized_input == True
        x_normalized = x
        target_normalized = self.normalize_target(target, dset_idx)

        out, td_data = self.forward(x_normalized)

        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict = self.get_reconstruction_loss(out,
                                                        target_normalized,
                                                        x_normalized,
                                                        dset_idx,
                                                        loss_idx,
                                                        return_predicted_img=False)

        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        recons_loss = self.split_w * recons_loss_dict['loss']
        if self.non_stochastic_version:
            kl_loss = torch.Tensor([0.0]).cuda()
            net_loss = recons_loss
        else:
            kl_loss = self.get_kl_divergence_loss(td_data)
            net_loss = recons_loss + self.get_kl_weight() * kl_loss

        mask = loss_idx == LossType.Elbo
        # if 2 * target_normalized.shape[1] == out.shape[1]:
        #     pred_mean, pred_logvar = out.chunk(2, dim=1)
        assert target_normalized.shape[1] == out.shape[1]
        mixed_loss = None
        if (~mask).sum() > 0:
            pred_x_normalized, _ = self.get_mixed_prediction(out[~mask], None, dset_idx[~mask])
            mixed_recons_ll = self.likelihood_finetuning.log_likelihood(x_normalized[~mask], {
                'mean': pred_x_normalized,
                'logvar': None
            })
            mixed_loss = (-1 * mixed_recons_ll).mean()
            net_loss += self.mixed_rec_w * mixed_loss

        if enable_logging:
            for i, x in enumerate(td_data['debug_qvar_max']):
                self.log(f'qvar_max:{i}', x.item(), on_epoch=True)

            self.log('reconstruction_loss', recons_loss_dict['loss'], on_epoch=True)
            self.log('kl_loss', kl_loss, on_epoch=True)
            self.log('training_loss', net_loss, on_epoch=True)
            self.log('lr', self.lr, on_epoch=True)
            if mixed_loss is not None:
                self.log('mixed_loss', mixed_loss)
            # self.log('grad_norm_bottom_up', self.grad_norm_bottom_up, on_epoch=True)
            # self.log('grad_norm_top_down', self.grad_norm_top_down, on_epoch=True)

        output = {
            'loss': net_loss,
            'reconstruction_loss': recons_loss.detach() if isinstance(recons_loss, torch.Tensor) else recons_loss,
            'kl_loss': kl_loss.detach(),
        }
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output

    def set_params_to_same_device_as(self, correct_device_tensor):
        self.likelihood.set_params_to_same_device_as(correct_device_tensor)
        self.likelihood_finetuning.set_params_to_same_device_as(correct_device_tensor)
        for dataset_index in [0, 1]:
            str_idx = f'subdset_{dataset_index}'
            if str_idx in self.data_mean and isinstance(self.data_mean[str_idx]['target'], torch.Tensor):
                if self.data_mean[str_idx]['target'].device != correct_device_tensor.device:
                    self.data_mean[str_idx]['target'] = self.data_mean[str_idx]['target'].to(
                        correct_device_tensor.device)
                    self.data_std[str_idx]['target'] = self.data_std[str_idx]['target'].to(correct_device_tensor.device)

                    self.data_mean[str_idx]['input'] = self.data_mean[str_idx]['input'].to(correct_device_tensor.device)
                    self.data_std[str_idx]['input'] = self.data_std[str_idx]['input'].to(correct_device_tensor.device)

    def validation_step(self, batch, batch_idx):
        x, target = batch[:2]
        dset_idx = torch.zeros((x.shape[0], ), dtype=torch.long).to(x.device)
        loss_idx = torch.Tensor([LossType.Elbo] * x.shape[0]).type(torch.long).to(x.device)
        self.set_params_to_same_device_as(target)

        x_normalized = x
        target_normalized = self.normalize_target(target, dset_idx)
        assert self.reconstruction_mode is False

        out, td_data = self.forward(x_normalized)
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict, recons_img = self.get_reconstruction_loss(out,
                                                                    target_normalized,
                                                                    x_normalized,
                                                                    dset_idx,
                                                                    loss_idx,
                                                                    return_predicted_img=True)

        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        channels_rinvpsnr = []
        for i in range(recons_img.shape[1]):
            self.channels_psnr[i].update(recons_img[:, i], target_normalized[:, i])
            psnr = RangeInvariantPsnr(target_normalized[:, i].clone(), recons_img[:, i].clone())
            channels_rinvpsnr.append(psnr)
            psnr = torch_nanmean(psnr).item()
            self.log(f'val_psnr_l{i+1}', psnr, on_epoch=True)

        recons_loss = recons_loss_dict['loss']
        # kl_loss = self.get_kl_divergence_loss(td_data)
        # net_loss = recons_loss + self.get_kl_weight() * kl_loss
        self.log('val_loss', recons_loss, on_epoch=True)

        # if batch_idx == 0 and self.power_of_2(self.current_epoch):
        #     all_samples = []
        #     for i in range(20):
        #         sample, _ = self(x_normalized[0:1, ...])
        #         sample = self.likelihood.get_mean_lv(sample)[0]
        #         all_samples.append(sample[None])

        #     all_samples = torch.cat(all_samples, dim=0)
        #     data_mean, data_std = self.get_mean_std_for_one_batch(dset_idx, self.data_mean, self.data_std)
        #     all_samples = all_samples * data_std['target'] + data_mean['target']
        #     all_samples = all_samples.cpu()
        #     img_mmse = torch.mean(all_samples, dim=0)[0]
        #     self.log_images_for_tensorboard(all_samples[:, 0, 0, ...], target[0, 0, ...], img_mmse[0], 'label1')
        #     self.log_images_for_tensorboard(all_samples[:, 0, 1, ...], target[0, 1, ...], img_mmse[1], 'label2')


if __name__ == '__main__':
    import numpy as np
    import torch

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

    # dset_idx = torch.Tensor([0, 0, 0, 1, 1, 0])

    # mean, std = LadderVaeTwoDset.get_mean_std_for_one_batch(dset_idx, data_mean, data_std)

    # from denoisplit.configs.microscopy_multi_channel_lvae_config import get_config
    from denoisplit.configs.twodset_config import get_config
    config = get_config()
    model = LadderVaeTwoDsetFinetuning(data_mean, data_std, config)
    mc = 1 if config.data.multiscale_lowres_count is None else config.data.multiscale_lowres_count
    inp = torch.rand((2, mc, config.data.image_size, config.data.image_size))
    out, td_data = model(inp)
    batch = (
        torch.rand((16, mc, config.data.image_size, config.data.image_size)),
        torch.rand((16, 2, config.data.image_size, config.data.image_size)),
        (torch.rand((16, )) > 0.5).type(torch.long),
        torch.Tensor([LossType.Elbo] * 8 + [LossType.ElboMixedReconstruction] * 8).type(torch.long),
    )
    model.validation_step(batch, 0)
    model.training_step(batch, 0)
