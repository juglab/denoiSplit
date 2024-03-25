"""
Multi dataset based setup.
"""
import torch
import torch.nn as nn

from denoisplit.core.loss_type import LossType
from denoisplit.core.psnr import RangeInvariantPsnr
from denoisplit.loss.exclusive_loss import compute_exclusion_loss
from denoisplit.loss.restricted_reconstruction_loss import RestrictedReconstruction
from denoisplit.nets.lvae import LadderVAE, compute_batch_mean, torch_nanmean


class LadderVaeTwoDsetRestrictedRecons(LadderVAE):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at, target_ch)
        self.automatic_optimization = False
        assert config.loss.loss_type == LossType.ElboRestrictedReconstruction, "This model only supports ElboRestrictedReconstruction loss type."
        self._interchannel_weights = None
        self.split_w = config.loss.split_weight

        if config.model.get('enable_learnable_interchannel_weights', False):
            # self._interchannel_weights = nn.Parameter(torch.ones((1, target_ch, 1, 1)), requires_grad=True)
            self._interchannel_weights = nn.Conv2d(target_ch, target_ch, 1, bias=True, groups=target_ch)
            self._interchannel_weights.weight.data.fill_(1.0 * 0.01)
            self._interchannel_weights.bias.data.fill_(0.0)

        self.init_normalization(data_mean, data_std)
        self.rest_recons_loss = RestrictedReconstruction(1, self.mixed_rec_w)
        # self.rest_recons_loss.update_only_these_till_kth_epoch(
        #     ['_interchannel_weights.weight', '_interchannel_weights.bias'], 40)

        print(f'[{self.__class__.__name__}] Learnable Ch weights:', self._interchannel_weights is not None)

    def init_normalization(self, data_mean, data_std):
        for dloader_key in self.data_mean.keys():
            assert dloader_key in ['subdset_0', 'subdset_1']
            for data_key in self.data_mean[dloader_key].keys():
                assert data_key in ['target', 'input']
                self.data_mean[dloader_key][data_key] = torch.Tensor(data_mean[dloader_key][data_key])
                self.data_std[dloader_key][data_key] = torch.Tensor(data_std[dloader_key][data_key])

            self.data_mean[dloader_key]['input'] = self.data_mean[dloader_key]['input'].reshape(1, 1, 1, 1)
            self.data_std[dloader_key]['input'] = self.data_std[dloader_key]['input'].reshape(1, 1, 1, 1)

    def get_reconstruction_loss(self,
                                reconstruction,
                                target,
                                input,
                                dset_idx,
                                loss_type_idx,
                                return_predicted_img=False,
                                likelihood_obj=None):
        output = self._get_reconstruction_loss_vector(reconstruction,
                                                      target,
                                                      input,
                                                      dset_idx,
                                                      return_predicted_img=return_predicted_img,
                                                      likelihood_obj=likelihood_obj)
        loss_dict = output[0] if return_predicted_img else output
        individual_ch_loss_mask = loss_type_idx == LossType.Elbo
        if torch.sum(individual_ch_loss_mask) > 0:
            loss_dict['loss'] = torch.mean(loss_dict['loss'][individual_ch_loss_mask])
            loss_dict['ch1_loss'] = torch.mean(loss_dict['ch1_loss'][individual_ch_loss_mask])
            loss_dict['ch2_loss'] = torch.mean(loss_dict['ch2_loss'][individual_ch_loss_mask])
        else:
            loss_dict['loss'] = 0.0
            loss_dict['ch1_loss'] = 0.0
            loss_dict['ch2_loss'] = 0.0

        if return_predicted_img:
            assert len(output) == 2
            return loss_dict, output[1]
        else:
            return loss_dict

    def normalize_target(self, target, dataset_index):
        dataset_index = dataset_index[:, None, None, None]
        mean0 = self.data_mean['subdset_0']['target']
        mean1 = self.data_mean['subdset_1']['target']
        std0 = self.data_std['subdset_0']['target']
        std1 = self.data_std['subdset_1']['target']

        mean = mean0 * (1 - dataset_index) + mean1 * dataset_index
        std = std0 * (1 - dataset_index) + std1 * dataset_index
        return (target - mean) / std

    def _get_reconstruction_loss_vector(self,
                                        reconstruction,
                                        target,
                                        input,
                                        dset_idx,
                                        return_predicted_img=False,
                                        likelihood_obj=None):
        """
        Args:
            return_predicted_img: If set to True, the besides the loss, the reconstructed image is also returned.
        """

        output = {
            'loss': None,
            'mixed_loss': None,
        }
        for i in range(1, 1 + target.shape[1]):
            output['ch{}_loss'.format(i)] = None

        if likelihood_obj is None:
            likelihood_obj = self.likelihood
        # Log likelihood
        ll, like_dict = likelihood_obj(reconstruction, target)
        ll = self._get_weighted_likelihood(ll)
        if self.skip_nboundary_pixels_from_loss is not None and self.skip_nboundary_pixels_from_loss > 0:
            pad = self.skip_nboundary_pixels_from_loss
            ll = ll[:, :, pad:-pad, pad:-pad]
            like_dict['params']['mean'] = like_dict['params']['mean'][:, :, pad:-pad, pad:-pad]

        assert ll.shape[1] == 2, f"Change the code below to handle >2 channels first. ll.shape {ll.shape}"
        output = {
            'loss': compute_batch_mean(-1 * ll),
        }
        if ll.shape[1] > 1:
            for i in range(1, 1 + target.shape[1]):
                output['ch{}_loss'.format(i)] = compute_batch_mean(-ll[:, i - 1])
        else:
            assert ll.shape[1] == 1
            output['ch1_loss'] = output['loss']
            output['ch2_loss'] = output['loss']

        if self.channel_1_w is not None or self.channel_2_w is not None:
            assert ll.shape[1] == 2, "Only 2 channels are supported for now."
            output['loss'] = (self.channel_1_w * output['ch1_loss'] +
                              self.channel_2_w * output['ch2_loss']) / (self.channel_1_w + self.channel_2_w)

            # if self._multiscale_count is not None and self._multiscale_count > 1:
            #     assert input.shape[1] == self._multiscale_count
            #     input = input[:, :1]

            # assert input.shape == mixed_pred.shape, "No fucking room for vectorization induced bugs."
            # mixed_recons_ll = self.likelihood.log_likelihood(input, {'mean': mixed_pred, 'logvar': mixed_logvar})
            # output['mixed_loss'] = compute_batch_mean(-1 * mixed_recons_ll)

        if return_predicted_img:
            return output, like_dict['params']['mean']

        return output

    @staticmethod
    def get_mean_std_for_one_batch(dset_idx, data_mean, data_std):
        """
        For each element in the batch, pick the relevant mean and stdev on the basis of which dataset it is coming from.
        """
        # to make it work as an index
        dset_idx = dset_idx.type(torch.long)
        batch_data_mean = {}
        batch_data_std = {}
        for key in data_mean['subdset_0'].keys():
            assert key in ['target', 'input']
            combined = torch.cat([data_mean['subdset_0'][key], data_mean['subdset_1'][key]], dim=0)
            batch_values = combined[dset_idx]
            batch_data_mean[key] = batch_values
            combined = torch.cat([data_std['subdset_0'][key], data_std['subdset_1'][key]], dim=0)
            batch_values = combined[dset_idx]
            batch_data_std[key] = batch_values

        return batch_data_mean, batch_data_std

    def get_mixed_prediction(self, prediction_mean, prediction_logvar, dset_idx):
        data_mean, data_std = self.get_mean_std_for_one_batch(dset_idx, self.data_mean, self.data_std)
        # NOTE: We should not have access to target data_mean, data_std of the dataset2. We should have access to
        # input data_mean, data_std of the dataset2.
        data_mean['target'] = self.data_mean['subdset_0']['target']
        data_std['target'] = self.data_std['subdset_0']['target']

        # NOTE: here, we are using the same interchannel weights for both dataset types. However,
        # we filter the loss on entries in get_reconstruction_loss()
        if self._interchannel_weights is not None:
            prediction_mean = self._interchannel_weights(prediction_mean)

        mixed_pred, mixed_logvar = super().get_mixed_prediction(prediction_mean,
                                                                prediction_logvar,
                                                                data_mean,
                                                                data_std,
                                                                channel_weights=None)
        return mixed_pred, mixed_logvar

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target, dset_idx, loss_idx = batch
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
        if self.non_stochastic_version:
            kl_loss = torch.Tensor([0.0]).cuda()
            net_loss = recons_loss
        else:
            kl_loss = self.get_kl_divergence_loss(td_data)
            net_loss = recons_loss + self.get_kl_weight() * kl_loss

        mask = loss_idx == LossType.Elbo
        exclusion_loss = None
        if self._exclusion_loss_weight > 0 and torch.sum(~mask) > 0:
            exclusion_loss = compute_exclusion_loss(out[~mask, 0], out[~mask, 1])
            net_loss += exclusion_loss * self._exclusion_loss_weight

        if isinstance(net_loss, torch.Tensor):
            self.manual_backward(net_loss, retain_graph=True)
        else:
            assert net_loss == 0.0
            return None

        assert self.loss_type == LossType.ElboRestrictedReconstruction
        if 2 * target_normalized.shape[1] == out.shape[1]:
            pred_mean, pred_logvar = out.chunk(2, dim=1)
        pred_x_normalized, _ = self.get_mixed_prediction(pred_mean[~mask], pred_logvar[~mask], dset_idx[~mask])
        params = list(self.named_parameters())
        loss_dict = self.rest_recons_loss.update_gradients(params, x_normalized[~mask], target_normalized[mask],
                                                           pred_mean[mask], pred_x_normalized, self.current_epoch)
        optim.step()
        if enable_logging:
            if exclusion_loss is not None:
                self.log('exclusive_loss', exclusion_loss.item(), on_epoch=True)

            for i, x in enumerate(td_data['debug_qvar_max']):
                self.log(f'qvar_max:{i}', x.item(), on_epoch=True)

            self.log('reconstruction_loss', recons_loss_dict['loss'], on_epoch=True)
            self.log('kl_loss', kl_loss, on_epoch=True)
            self.log('training_loss', net_loss, on_epoch=True)
            self.log('lr', self.lr, on_epoch=True)
            if self._interchannel_weights is not None:
                self.log('interchannel_w0',
                         self._interchannel_weights.weight.squeeze()[0].item(),
                         on_epoch=False,
                         on_step=True)
                self.log('interchannel_w1',
                         self._interchannel_weights.weight.squeeze()[1].item(),
                         on_epoch=False,
                         on_step=True)
                if self._interchannel_weights.bias is not None:
                    self.log('interchannel_b0',
                             self._interchannel_weights.bias.squeeze()[0].item(),
                             on_epoch=False,
                             on_step=True)
                    self.log('interchannel_b1',
                             self._interchannel_weights.bias.squeeze()[1].item(),
                             on_epoch=False,
                             on_step=True)

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
        if isinstance(self._interchannel_weights, torch.Tensor):
            if self._interchannel_weights.device != correct_device_tensor.device:
                self._interchannel_weights = self._interchannel_weights.to(correct_device_tensor.device)

        for dataset_index in [0, 1]:
            str_idx = f'subdset_{dataset_index}'
            if str_idx in self.data_mean and isinstance(self.data_mean[str_idx]['target'], torch.Tensor):
                if self.data_mean[str_idx]['target'].device != correct_device_tensor.device:
                    self.data_mean[str_idx]['target'] = self.data_mean[str_idx]['target'].to(
                        correct_device_tensor.device)
                    self.data_std[str_idx]['target'] = self.data_std[str_idx]['target'].to(correct_device_tensor.device)

                    self.data_mean[str_idx]['input'] = self.data_mean[str_idx]['input'].to(correct_device_tensor.device)
                    self.data_std[str_idx]['input'] = self.data_std[str_idx]['input'].to(correct_device_tensor.device)

                    self.likelihood.set_params_to_same_device_as(correct_device_tensor)
                else:
                    return

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
    import numpy as np
    import torch

    # from denoisplit.configs.microscopy_multi_channel_lvae_config import get_config
    from denoisplit.configs.twodset_config import get_config
    config = get_config()
    model = LadderVaeTwoDsetRestrictedRecons(data_mean, data_std, config)
    mc = 1 if config.data.multiscale_lowres_count is None else config.data.multiscale_lowres_count
    inp = torch.rand((2, mc, config.data.image_size, config.data.image_size))
    out, td_data = model(inp)
    batch = (
        torch.rand((16, mc, config.data.image_size, config.data.image_size)),
        torch.rand((16, 2, config.data.image_size, config.data.image_size)),
        (torch.rand((16, )) > 0.5).type(torch.long),
        torch.Tensor([LossType.Elbo] * 8 + [LossType.ElboMixedReconstruction] * 8).type(torch.long),
    )
    model.training_step(batch, 0)
    model.validation_step(batch, 0)
