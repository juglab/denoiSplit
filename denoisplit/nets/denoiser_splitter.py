import os
from copy import deepcopy

import torch

import ml_collections
from denoisplit.config_utils import load_config
from denoisplit.core.loss_type import LossType
from denoisplit.nets.lvae import LadderVAE, RangeInvariantPsnr, torch_nanmean
from denoisplit.nets.lvae_denoiser import LadderVAEDenoiser


class DenoiserSplitter(LadderVAE):
    """
    It denoises the input and optionally the target. And then it splits the denoised input.
    """

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        self._denoiser_mmse = config.model.get('denoiser_mmse', 1)
        self._denoiser_kinput_samples = config.model.get('denoiser_kinput_samples', None)
        if self._denoiser_kinput_samples is not None:
            assert self._denoiser_kinput_samples >= 1
            assert self._denoiser_mmse == 1

        self._synchronized_input_target = config.model.get('synchronized_input_target', False)
        self._use_noisy_input = config.model.get('use_noisy_input', False)
        self._use_noisy_target = config.model.get('use_noisy_target', False)
        self._use_both_noisy_clean_input = config.model.get('use_both_noisy_clean_input', False)

        new_config = deepcopy(ml_collections.ConfigDict(config))
        with new_config.unlocked():
            new_config.data.image_size = new_config.data.image_size // 2
            if self._use_both_noisy_clean_input:
                new_config.data.color_ch = new_config.data.get('color_ch', 1) + 1
            if self._denoiser_kinput_samples is not None:
                new_config.data.color_ch += (self._denoiser_kinput_samples - 1)
        super().__init__(data_mean, data_std, new_config, use_uncond_mode_at, target_ch)

        self._denoiser_ch1, config_ch1 = self.load_denoiser(config.model.get('pre_trained_ckpt_fpath_ch1', None))
        self._denoiser_ch2, config_ch2 = self.load_denoiser(config.model.get('pre_trained_ckpt_fpath_ch2', None))
        self._denoiser_input, config_inp = self.load_denoiser(config.model.get('pre_trained_ckpt_fpath_input', None))
        self._denoiser_all, config_all = self.load_denoiser(config.model.get('pre_trained_ckpt_fpath_all', None))

        # Same noise level for all denoisers
        if 'synthetic_gaussian_scale' in config.data:
            assert config_ch1 is None or ('synthetic_gaussian_scale' in config_ch1.data
                                          and config_ch1.data.synthetic_gaussian_scale
                                          == config.data.synthetic_gaussian_scale)
            assert config_ch2 is None or ('synthetic_gaussian_scale' in config_ch2.data
                                          and config_ch2.data.synthetic_gaussian_scale
                                          == config.data.synthetic_gaussian_scale)
            assert config_inp is None or ('synthetic_gaussian_scale' in config_inp.data
                                          and config_inp.data.synthetic_gaussian_scale
                                          == config.data.synthetic_gaussian_scale)
            assert config_all is None or ('synthetic_gaussian_scale' in config_all.data
                                          and config_all.data.synthetic_gaussian_scale
                                          == config.data.synthetic_gaussian_scale)

        if self._denoiser_all is not None:
            self._denoiser_ch1 = self._denoiser_all
            self._denoiser_ch2 = self._denoiser_all
            self._denoiser_input = self._denoiser_all
        else:
            if self._denoiser_ch1 is not None:
                idx = ['Ch1', 'Ch2'].index(self._denoiser_ch1.denoise_channel)
                fname = config_ch1.data[f'ch{idx+1}_fname']
                assert config.data['ch1_fname'] == fname
            if self._denoiser_ch2 is not None:
                idx = ['Ch1', 'Ch2'].index(self._denoiser_ch2.denoise_channel)
                fname = config_ch2.data[f'ch{idx+1}_fname']
                assert config.data['ch2_fname'] == fname

        den_ch1 = self._denoiser_ch1 is not None
        den_ch2 = self._denoiser_ch2 is not None
        den_input = self._denoiser_input is not None
        assert self._denoiser_input is None or (self._use_noisy_input == False
                                                or self._use_both_noisy_clean_input == True)
        print(f'[{self.__class__}] Denoisers Ch1:{den_ch1}, Ch2:{den_ch2}, Input:{den_input} All:{den_input}')

    def load_data_mean_std(self, checkpoint):
        # TODO: save the mean and std in the checkpoint.
        data_mean = deepcopy(self.data_mean)
        data_std = deepcopy(self.data_std)
        return data_mean, data_std

    def load_denoiser(self, pre_trained_ckpt_fpath):
        if pre_trained_ckpt_fpath is None:
            return None, None
        checkpoint = torch.load(pre_trained_ckpt_fpath)
        config_fpath = os.path.join(os.path.dirname(pre_trained_ckpt_fpath), 'config.pkl')
        config = load_config(config_fpath)
        data_mean, data_std = self.load_data_mean_std(checkpoint)

        model = LadderVAEDenoiser(data_mean, data_std, config)
        _ = model.load_state_dict(checkpoint['state_dict'], strict=True)
        print('Loaded model from ckpt dir', pre_trained_ckpt_fpath, f' at epoch:{checkpoint["epoch"]}')

        for param in model.parameters():
            param.requires_grad = False
        return model, config

    def denoise_one_channel(self, normalized_x, denoiser, mmse_count=1, k_samples=None):
        if k_samples is None:
            output = 0
            for i in range(mmse_count):
                out, _ = denoiser(normalized_x)
                output += denoiser.likelihood.distr_params(out)['mean']
            return output / mmse_count
        else:
            output = []
            for i in range(k_samples):
                out, _ = denoiser(normalized_x)
                output.append(denoiser.likelihood.distr_params(out)['mean'])
            # batch * k_samples * ch * H * W
            return output

    def trim_to_half(self, x):
        H = x.shape[-1] // 2
        return x[:, :, H // 2:-H // 2, H // 2:-H // 2]

    def denoise_target(self, target_normalized):
        ch1 = target_normalized[:, :1]
        ch2 = target_normalized[:, 1:]
        ch1_denoised = self.denoise_one_channel(ch1, self._denoiser_ch1, mmse_count=self._denoiser_mmse)
        ch2_denoised = self.denoise_one_channel(ch2, self._denoiser_ch2, mmse_count=self._denoiser_mmse)

        ch1_denoised = self.trim_to_half(ch1_denoised)
        ch2_denoised = self.trim_to_half(ch2_denoised)
        return torch.cat([ch1_denoised, ch2_denoised], dim=1)

    def denoise_input(self, x_normalized):
        x_normalized = self.denoise_one_channel(x_normalized,
                                                self._denoiser_input,
                                                mmse_count=self._denoiser_mmse,
                                                k_samples=self._denoiser_kinput_samples)
        if self._denoiser_kinput_samples is not None:
            assert isinstance(x_normalized, list)
            return [self.trim_to_half(x) for x in x_normalized]
        return self.trim_to_half(x_normalized)

    def compute_input(self, target_normalized):
        return torch.mean(target_normalized, dim=1, keepdim=True)

    def get_normalized_input_target(self, batch):
        """
        Optionally denoise the input and target. For conssistency, we also trim them to half their spatial size.
        """
        x, noisy_target = batch[:2]
        noisy_target_normalized = self.normalize_target(noisy_target)
        denoised_target_normalized = self.denoise_target(noisy_target_normalized)

        if self._use_noisy_target:
            target_normalized = self.trim_to_half(noisy_target_normalized)
        else:
            target_normalized = denoised_target_normalized

        # inputs
        if self._use_both_noisy_clean_input:
            x_normalized = self.normalize_input(x)
            denoised_x = self.denoise_input(x_normalized)
            x_normalized = self.trim_to_half(x_normalized)
            assert isinstance(denoised_x, list)
            x_normalized = torch.cat([x_normalized] + denoised_x, dim=1)
        elif self._use_noisy_input:
            x_normalized = self.normalize_input(x)
            x_normalized = self.trim_to_half(x_normalized)
            assert self._synchronized_input_target != True
        elif self._synchronized_input_target:
            x_normalized = torch.mean(target_normalized, dim=1, keepdim=True)
        elif self._denoiser_input is not None:
            x_normalized = self.denoise_input(x)
        else:
            raise ValueError('Not clear how input needs to be computed.')
        return x_normalized, target_normalized

    def training_step(self, batch, batch_idx, enable_logging=True):
        x_normalized, target_normalized = self.get_normalized_input_target(batch)
        out, td_data = self.forward(x_normalized)
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict, imgs = self.get_reconstruction_loss(out,
                                                              target_normalized,
                                                              x_normalized,
                                                              return_predicted_img=True)

        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        recons_loss = recons_loss_dict['loss']
        if self.loss_type == LossType.ElboMixedReconstruction:
            recons_loss += self.mixed_rec_w * recons_loss_dict['mixed_loss']
            if enable_logging:
                self.log('mixed_reconstruction_loss', recons_loss_dict['mixed_loss'], on_epoch=True)
        elif self.loss_type == LossType.ElboWithNbrConsistency:
            assert len(batch) == 4
            grid_sizes = batch[-1]
            nbr_cons_loss = self.nbr_consistency_w * self.nbr_consistency_loss.get(imgs, grid_sizes=grid_sizes)
            # print(recons_loss, nbr_cons_loss)
            self.log('nbr_cons_loss', nbr_cons_loss.item(), on_epoch=True)
            recons_loss += nbr_cons_loss

        if self.non_stochastic_version:
            kl_loss = torch.Tensor([0.0]).cuda()
            net_loss = recons_loss
        else:
            kl_loss = self.get_kl_divergence_loss(
                td_data) if self.kl_loss_formulation != 'usplit' else self.get_kl_divergence_loss_usplit(td_data)
            net_loss = recons_loss + self.get_kl_weight() * kl_loss

        if enable_logging:
            for i, x in enumerate(td_data['debug_qvar_max']):
                self.log(f'qvar_max:{i}', x.item(), on_epoch=True)

            self.log('reconstruction_loss', recons_loss_dict['loss'], on_epoch=True)
            self.log('kl_loss', kl_loss, on_epoch=True)
            self.log('training_loss', net_loss, on_epoch=True)
            self.log('lr', self.lr, on_epoch=True)
            # self.log('grad_norm_bottom_up', self.grad_norm_bottom_up, on_epoch=True)
            # self.log('grad_norm_top_down', self.grad_norm_top_down, on_epoch=True)

        output = {
            'loss': net_loss,
            'reconstruction_loss': recons_loss.detach(),
            'kl_loss': kl_loss.detach(),
        }
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output

    def validation_step(self, batch, batch_idx):
        self.set_params_to_same_device_as(batch[0])
        x_normalized, target_normalized = self.get_normalized_input_target(batch)

        out, td_data = self.forward(x_normalized)
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict, recons_img = self.get_reconstruction_loss(out,
                                                                    target_normalized,
                                                                    x_normalized,
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

        # self.label1_psnr.update(recons_img[:, 0], target_normalized[:, 0])
        # self.label2_psnr.update(recons_img[:, 1], target_normalized[:, 1])

        # psnr_label1 = RangeInvariantPsnr(target_normalized[:, 0].clone(), recons_img[:, 0].clone())
        # psnr_label2 = RangeInvariantPsnr(target_normalized[:, 1].clone(), recons_img[:, 1].clone())
        recons_loss = recons_loss_dict['loss']
        # kl_loss = self.get_kl_divergence_loss(td_data)
        # net_loss = recons_loss + self.get_kl_weight() * kl_loss
        self.log('val_loss', recons_loss, on_epoch=True)
        # val_psnr_l1 = torch_nanmean(psnr_label1).item()
        # val_psnr_l2 = torch_nanmean(psnr_label2).item()
        # self.log('val_psnr_l1', val_psnr_l1, on_epoch=True)
        # self.log('val_psnr_l2', val_psnr_l2, on_epoch=True)
        # self.log('val_psnr', (val_psnr_l1 + val_psnr_l2) / 2, on_epoch=True)

        # if batch_idx == 0 and self.power_of_2(self.current_epoch):
        #     all_samples = []
        #     for i in range(20):
        #         sample, _ = self(x_normalized[0:1, ...])
        #         sample = self.likelihood.get_mean_lv(sample)[0]
        #         all_samples.append(sample[None])

        #     all_samples = torch.cat(all_samples, dim=0)
        #     all_samples = all_samples * self.data_std['target'] + self.data_mean['target']
        #     all_samples = all_samples.cpu()
        #     img_mmse = torch.mean(all_samples, dim=0)[0]
        #     self.log_images_for_tensorboard(all_samples[:, 0, 0, ...], noisy_target[0, 0, ...], img_mmse[0], 'label1')
        #     self.log_images_for_tensorboard(all_samples[:, 0, 1, ...], noisy_target[0, 1, ...], img_mmse[1], 'label2')


if __name__ == '__main__':
    import numpy as np
    import torch

    from denoisplit.configs.denoiser_splitting_config import get_config

    config = get_config()
    data_mean = {'input': np.array([0]).reshape(1, 1, 1, 1), 'target': np.array([0, 0]).reshape(1, 2, 1, 1)}
    data_std = {'input': np.array([1]).reshape(1, 1, 1, 1), 'target': np.array([1, 1]).reshape(1, 2, 1, 1)}
    model = DenoiserSplitter(data_mean, data_std, config)
    mc = 1 if config.data.multiscale_lowres_count is None else config.data.multiscale_lowres_count + 1
    inp = torch.rand((2, mc, config.data.image_size, config.data.image_size))
    # out, td_data = model(inp)
    # print(out.shape)
    batch = (
        torch.rand((16, mc, config.data.image_size, config.data.image_size)),
        torch.rand((16, 2, config.data.image_size, config.data.image_size)),
    )
    model.training_step(batch, 0)
    model.validation_step(batch, 0)

    ll = torch.ones((12, 2, 32, 32))
    ll_new = model._get_weighted_likelihood(ll)
    print(ll_new[:, 0].mean(), ll_new[:, 0].std())
    print(ll_new[:, 1].mean(), ll_new[:, 1].std())
    print('mar')
