from distutils.command.config import LANG_EXT
from statistics import mode
from turtle import pd
from denoisplit.nets.lvae import LadderVAE, compute_batch_mean, torch_nanmean
import torch
from denoisplit.core.loss_type import LossType
from denoisplit.core.psnr import RangeInvariantPsnr
from denoisplit.loss.exclusive_loss import compute_exclusion_loss
from denoisplit.data_loader.pavia2_enums import Pavia2BleedthroughType
import torch.nn as nn


class LadderVAESemiSupervised(LadderVAE):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at, target_ch)
        assert self.enable_mixed_rec is True
        self._exclusion_loss_w = config.loss.get('exclusion_loss_weight', None)
        conv1 = nn.Conv2d(config.model.decoder.n_filters, 32, 5, stride=2, padding=2)
        conv2 = nn.Conv2d(32, 16, 5, stride=2, padding=2)
        conv3 = nn.Conv2d(16, 1, 5, stride=2, padding=2)
        self._factor_branch = nn.Sequential(conv1, nn.LeakyReLU(), conv2, nn.LeakyReLU(), conv3, nn.ReLU(),
                                            nn.AvgPool2d(8))
        print(f'[{self.__class__.__name__}] Exclusion Loss w', self._exclusion_loss_w)

    def get_factor(self, reconstruction):
        factor = self._factor_branch(reconstruction) + 1
        return factor

    def get_mixed_prediction(self, reconstruction, channelwise_prediction, channelwise_logvar):
        factor = self.get_factor(reconstruction)

        mixed_prediction = channelwise_prediction[:, :1] * factor + channelwise_prediction[:, 1:]

        var = torch.exp(channelwise_logvar)
        # sum of variance.
        var = var[:, :1] * (factor * factor) + var[:, 1:]
        logvar = torch.log(var)

        return mixed_prediction, logvar

    def _get_reconstruction_loss_vector(self, reconstruction, input, target_ch1, return_predicted_img=False):
        """
        Args:
            return_predicted_img: If set to True, the besides the loss, the reconstructed image is also returned.
        """

        # Log likelihood
        ll, like_dict = self.likelihood(reconstruction, target_ch1)

        # We just want to compute it for the first channel.
        ll = ll[:, :1]

        if self.skip_nboundary_pixels_from_loss is not None and self.skip_nboundary_pixels_from_loss > 0:
            pad = self.skip_nboundary_pixels_from_loss
            ll = ll[:, :, pad:-pad, pad:-pad]
            like_dict['params']['mean'] = like_dict['params']['mean'][:, :, pad:-pad, pad:-pad]

        recons_loss = compute_batch_mean(-1 * ll)
        exclusion_loss = None
        if self._exclusion_loss_w:
            exclusion_loss = compute_exclusion_loss(reconstruction[:, :1], reconstruction[:, 1:])

        output = {
            'loss': recons_loss,
            'ch1_loss': compute_batch_mean(-ll[:, 0]),
            'ch2_loss': None,
            'exclusion_loss': exclusion_loss
        }

        mixed_target = input[:, :1]
        mixed_prediction, mixed_logvar = self.get_mixed_prediction(reconstruction, like_dict['params']['mean'],
                                                                   like_dict['params']['logvar'])

        # TODO: We must enable standard deviation here in some way. I think this is very much needed.
        mixed_recons_ll = self.likelihood.log_likelihood(mixed_target, {
            'mean': mixed_prediction,
            'logvar': mixed_logvar
        })
        output['mixed_loss'] = compute_batch_mean(-1 * mixed_recons_ll)

        if return_predicted_img:
            return output, torch.cat([like_dict['params']['mean'], mixed_prediction], dim=1)

        return output

    def get_reconstruction_loss(self, reconstruction, input, target_ch1, return_predicted_img=False):
        output = self._get_reconstruction_loss_vector(reconstruction,
                                                      input,
                                                      target_ch1,
                                                      return_predicted_img=return_predicted_img)
        loss_dict = output[0] if return_predicted_img else output
        loss_dict['loss'] = torch.mean(loss_dict['loss'])
        loss_dict['ch1_loss'] = torch.mean(loss_dict['ch1_loss'])
        loss_dict['ch2_loss'] = None

        if 'mixed_loss' in loss_dict:
            loss_dict['mixed_loss'] = torch.mean(loss_dict['mixed_loss'])
        if return_predicted_img:
            assert len(output) == 2
            return loss_dict, output[1]
        else:
            return loss_dict

    def normalize_target(self, target, dataset_index):
        mean_ = self.data_mean[dataset_index, :, 1:]
        assert mean_.shape[-1] == 1
        mean_ = mean_[..., 0]
        assert len(mean_) == len(target)
        std_ = self.data_std[dataset_index, :, 1:]
        return (target - mean_) / std_[..., 0]

    def normalize_input(self, x, dataset_index):
        if self.normalized_input:
            return x
        return (x - self.data_mean[dataset_index].mean()) / self.data_std[dataset_index].mean()

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target, dataset_index = batch
        x_normalized = self.normalize_input(x, dataset_index)
        target_normalized = self.normalize_target(target, dataset_index)

        out, td_data = self.forward(x_normalized)
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict = self.get_reconstruction_loss(out,
                                                        x_normalized,
                                                        target_normalized,
                                                        return_predicted_img=False)

        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        recons_loss = recons_loss_dict['loss']
        assert self.loss_type == LossType.ElboSemiSupMixedReconstruction

        recons_loss += self.mixed_rec_w * recons_loss_dict['mixed_loss']

        if enable_logging:
            self.log('mixed_reconstruction_loss', recons_loss_dict['mixed_loss'], on_epoch=True)

        kl_loss = self.get_kl_divergence_loss(td_data)
        net_loss = recons_loss + self.get_kl_weight() * kl_loss
        if self._exclusion_loss_w:
            excl_loss = self._exclusion_loss_w * recons_loss_dict['exclusion_loss']
            net_loss += net_loss
            if enable_logging:
                self.log('exclusion_loss', excl_loss, on_epoch=True)

        if enable_logging:
            for i, x in enumerate(td_data['debug_qvar_max']):
                self.log(f'qvar_max:{i}', x.item(), on_epoch=True)

            self.log('reconstruction_loss', recons_loss_dict['loss'], on_epoch=True)
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

    def validation_step(self, batch, batch_idx):
        x, target, dataset_index = batch
        self.set_params_to_same_device_as(target)

        x_normalized = self.normalize_input(x, dataset_index)
        target_normalized = self.normalize_target(target, dataset_index)

        out, td_data = self.forward(x_normalized)

        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict, recons_img = self.get_reconstruction_loss(out,
                                                                    x_normalized,
                                                                    target_normalized,
                                                                    return_predicted_img=True)
        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        self.label1_psnr.update(recons_img[:, 0], target_normalized[:, 0])
        psnr_label1 = RangeInvariantPsnr(target_normalized[:, 0].clone(), recons_img[:, 0].clone())
        recons_loss = recons_loss_dict['loss']
        self.log('val_loss', recons_loss, on_epoch=True)
        val_psnr_l1 = torch_nanmean(psnr_label1).item()
        self.log('val_psnr_l1', val_psnr_l1, on_epoch=True)

        if batch_idx == 0 and self.power_of_2(self.current_epoch):
            all_samples = []
            for i in range(20):
                sample, _ = self(x_normalized[0:1, ...])
                sample = self.likelihood.get_mean_lv(sample)[0]
                all_samples.append(sample[None])

            all_samples = torch.cat(all_samples, dim=0)
            all_samples = all_samples * self.data_std[dataset_index[0]] + self.data_mean[dataset_index[0]]
            all_samples = all_samples.cpu()
            img_mmse = torch.mean(all_samples, dim=0)[0]
            self.log_images_for_tensorboard(all_samples[:, 0, 0, ...], target[0, 0, ...], img_mmse[0], 'label1')

    def on_validation_epoch_end(self):
        psnrl1 = self.label1_psnr.get()
        psnr = psnrl1
        self.log('val_psnr', psnr, on_epoch=True)
        self.label1_psnr.reset()


if __name__ == '__main__':
    from denoisplit.configs.semi_supervised_config import get_config
    config = get_config()
    data_mean = torch.ones([3, 1, 2, 1, 1])
    data_std = torch.ones([3, 1, 2, 1, 1])
    model = LadderVAESemiSupervised(data_mean, data_std, config)
    inp = torch.rand((32, 1, 64, 64))
    tar = torch.rand(32, 1, 64, 64)
    dset_index = torch.randint(low=0, high=3, size=(len(inp), ))
    model.training_step((inp, tar, dset_index), 0)
