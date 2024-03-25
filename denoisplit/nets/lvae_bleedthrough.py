"""
This model is created to handle the bleedthrough effect.
"""
from distutils.command.config import config

from numpy import dtype
from denoisplit.nets.lvae import LadderVAE, compute_batch_mean, torch_nanmean
import torch
from denoisplit.core.loss_type import LossType
from denoisplit.core.psnr import RangeInvariantPsnr
from denoisplit.data_loader.pavia2_enums import Pavia2BleedthroughType


def empty_tensor(tens):
    """
    Returns true if there are no elements in this tensor. 
    """
    return tens.nelement() == 0


class LadderVAEWithMixedRecons(LadderVAE):
    """
    Ex: Pavia2 dataset.
    Here, we work with 2 data sources. For one data source, we have both channels. 
    For the other, we just have one channel and the input. Here, we apply the mixed reconstruction loss.

    """

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=3):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at=use_uncond_mode_at, target_ch=target_ch)
        assert isinstance(self.data_mean, dict)
        self.data_mean['target'] = torch.Tensor(self.data_mean['target'])
        self.data_mean['mix'] = torch.Tensor(self.data_mean['mix'])

        self.data_std['target'] = torch.Tensor(self.data_std['target'])
        self.data_std['mix'] = torch.Tensor(self.data_std['mix'])
        self.rec_loss_ch_w = config.loss.get('rec_loss_channel_weights',None)
        print(f'[{self.__class__.__name__}] Ch weights: {self.rec_loss_ch_w}')

    def normalize_input(self, x):
        if self.normalized_input:
            return x
        return (x - self.data_mean['mix']) / self.data_std['mix']

    def normalize_target(self, target):
        return (target - self.data_mean['target']) / self.data_std['target']

    def get_reconstruction_loss(self, reconstruction, input, target, return_predicted_img=False):
        if empty_tensor(reconstruction):
            return None, None

        output = self._get_reconstruction_loss_vector(reconstruction,
                                                      input,
                                                      target,
                                                      return_predicted_img=return_predicted_img)
        loss_dict = output[0] if return_predicted_img else output

        if return_predicted_img:
            assert len(output) == 2
            return loss_dict, output[1]
        else:
            return loss_dict

    def get_mixed_prediction(self, prediction, prediction_logvar):

        pred_unorm = prediction * self.data_std['target'] + self.data_mean['target']

        mixed_prediction = (torch.sum(pred_unorm, dim=1, keepdim=True) - self.data_mean['mix']) / self.data_std['mix']

        var = torch.exp(prediction_logvar)
        var = var * (self.data_std['target'] / self.data_std['mix'])**2
        # sum of variance.
        mixed_var = 0
        for i in range(var.shape[1]):
            mixed_var += var[:, i:i + 1]

        logvar = torch.log(mixed_var)

        return mixed_prediction, logvar

    def _get_reconstruction_loss_vector(self, reconstruction, input, target, return_predicted_img=False):
        """
        Args:
            return_predicted_img: If set to True, the besides the loss, the reconstructed image is also returned.
        """

        # Log likelihood
        ll, like_dict = self.likelihood(reconstruction, target)
        if self.skip_nboundary_pixels_from_loss is not None and self.skip_nboundary_pixels_from_loss > 0:
            pad = self.skip_nboundary_pixels_from_loss
            ll = ll[:, :, pad:-pad, pad:-pad]
            like_dict['params']['mean'] = like_dict['params']['mean'][:, :, pad:-pad, pad:-pad]

        recons_loss = compute_batch_mean(-1 * ll)
        output = {
            'loss': recons_loss if self.rec_loss_ch_w is None else 0,
        }
        for ch_idx in range(ll.shape[1]):
            ch_idx_loss = compute_batch_mean(-ll[:, ch_idx])
            output[f'ch{ch_idx}_loss'] = ch_idx_loss
            if self.rec_loss_ch_w is not None:
                assert len(self.rec_loss_ch_w) == ll.shape[1]
                output['loss'] += (self.rec_loss_ch_w[ch_idx] * ch_idx_loss)/sum(self.rec_loss_ch_w)
    

        
        assert self.enable_mixed_rec is True
        mixed_pred, mixed_logvar = self.get_mixed_prediction(like_dict['params']['mean'], like_dict['params']['logvar'])

        mixed_target = input
        mixed_recons_ll = self.likelihood.log_likelihood(mixed_target, {'mean': mixed_pred, 'logvar': mixed_logvar})
        output['mixed_loss'] = compute_batch_mean(-1 * mixed_recons_ll)

        if return_predicted_img:
            return output, like_dict['params']['mean']

        return output

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target, mixed_recons_flag = batch
        self.set_params_to_same_device_as(target)

        x_normalized = self.normalize_input(x)
        # TODO: check normalization. it is so because nucleus is from two datasets.
        target_normalized = self.normalize_target(target)
        out, td_data = self.forward(x_normalized)
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        clean_mask = mixed_recons_flag == Pavia2BleedthroughType.Clean
        recons_loss_dict, _ = self.get_reconstruction_loss(out,
                                                           x_normalized,
                                                           target_normalized,
                                                           return_predicted_img=True)

        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        channel_recons_loss = 0
        if recons_loss_dict is not None and clean_mask.sum() > 0:
            channel_recons_loss = torch.mean(recons_loss_dict['loss'][clean_mask])

        assert self.loss_type == LossType.ElboMixedReconstruction
        input_recons_loss = recons_loss_dict['mixed_loss'].mean()
        recons_loss = channel_recons_loss + self.mixed_rec_w * input_recons_loss

        kl_loss = self.get_kl_divergence_loss(td_data)
        net_loss = recons_loss + self.get_kl_weight() * kl_loss

        if enable_logging:
            self.log('mixed_reconstruction_loss', input_recons_loss, on_epoch=True)
            for i, x in enumerate(td_data['debug_qvar_max']):
                self.log(f'qvar_max:{i}', x.item(), on_epoch=True)
                self.log('kl_loss', kl_loss, on_epoch=True)
                self.log('grad_norm_bottom_up', self.grad_norm_bottom_up, on_epoch=True)
                self.log('grad_norm_top_down', self.grad_norm_top_down, on_epoch=True)
                self.log('lr', self.lr, on_epoch=True)
                if channel_recons_loss != 0:
                    self.log('channel_recons_loss', channel_recons_loss, on_epoch=True)
                self.log('input_recons_loss', input_recons_loss, on_epoch=True)
                self.log('training_loss', net_loss, on_epoch=True)

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
        x, target, _ = batch
        self.set_params_to_same_device_as(target)

        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)
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
            all_samples = all_samples * self.data_std['target'] + self.data_mean['target']
            all_samples = all_samples.cpu()
            img_mmse = torch.mean(all_samples, dim=0)[0]
            self.log_images_for_tensorboard(all_samples[:, 0, 0, ...], target[0, 0, ...], img_mmse[0], 'label1')
            self.log_images_for_tensorboard(all_samples[:, 0, 1, ...], target[0, 1, ...], img_mmse[1], 'label2')

    def set_params_to_same_device_as(self, correct_device_tensor):
        if isinstance(self.data_mean['mix'], torch.Tensor):
            if self.data_mean['mix'].device != correct_device_tensor.device:
                self.data_mean['mix'] = self.data_mean['mix'].to(correct_device_tensor.device)
                self.data_mean['target'] = self.data_mean['target'].to(correct_device_tensor.device)
                self.data_std['mix'] = self.data_std['mix'].to(correct_device_tensor.device)
                self.data_std['target'] = self.data_std['target'].to(correct_device_tensor.device)

                self.likelihood.set_params_to_same_device_as(correct_device_tensor)


if __name__ == '__main__':
    import numpy as np
    from denoisplit.configs.pavia2_config import get_config
    data_mean = {
        'target': np.array([0.0, 10.0], dtype=np.float32).reshape(1, 2, 1, 1),
        'mix': np.array([110.0], dtype=np.float32).reshape(1, 1, 1, 1),
    }
    data_std = {
        'target': np.array([1.0, 5], dtype=np.float32).reshape(1, 2, 1, 1),
        'mix': np.array([25.0], dtype=np.float32).reshape(1, 1, 1, 1),
    }
    config = get_config()
    model = LadderVAEWithMixedRecons(data_mean, data_std, config)
    x = torch.rand((32, 1, 64, 64), dtype=torch.float32)
    target = torch.rand((32, 2, 64, 64), dtype=torch.float32)
    mixed_recons_flag = torch.Tensor(np.array([1] * 32)).type(torch.bool)
    batch = (x, target, mixed_recons_flag)
    output = model.training_step(batch, 0)
    print('All ')