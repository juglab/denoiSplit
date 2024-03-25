from copy import deepcopy

import torch

import ml_collections
from denoisplit.nets.lvae import LadderVAE
from denoisplit.nets.lvae_twindecoder import LadderVAETwinDecoder


class LVAEWithDeepEncoder(LadderVAETwinDecoder):

    def __init__(self, data_mean, data_std, config):
        config = ml_collections.ConfigDict(config)
        new_config = deepcopy(config)
        with new_config.unlocked():
            new_config.data.color_ch = config.model.encoder.n_filters
            new_config.data.multiscale_lowres_count = None  # multiscaleing is inside the extra encoder.
            new_config.model.gated = False
            new_config.model.decoder.dropout = 0.
            new_config.model.merge_type = 'residual_ungated'
        super().__init__(data_mean, data_std, new_config)

        self.enable_input_alphasum_of_channels = config.data.target_separate_normalization == False
        with config.unlocked():
            config.model.non_stochastic_version = True
        self.extra_encoder = LadderVAE(data_mean, data_std, config, target_ch=config.model.encoder.n_filters)

    def forward(self, x):
        encoded, _ = self.extra_encoder(x)
        return super().forward(encoded)

    def normalize_target(self, target, batch=None):
        target_normalized = super().normalize_target(target)
        if self.enable_input_alphasum_of_channels:
            # adjust the targets for the alpha
            alpha = batch[2][:, None, None, None]
            tar1 = target_normalized[:, :1] * alpha
            tar2 = target_normalized[:, 1:] * (1 - alpha)
            target_normalized = torch.cat([tar1, tar2], dim=1)
        return target_normalized

    def training_step(self, batch, batch_idx):
        x, target = batch[:2]
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target, batch)
        if batch_idx == 0 and self.enable_input_alphasum_of_channels:
            assert torch.abs(torch.sum(target_normalized, dim=1, keepdim=True) -
                             x_normalized[:, :1]).max().item() < 1e-5

        out_l1, out_l2, td_data = self.forward(x_normalized)

        recons_loss = self.get_reconstruction_loss(out_l1, out_l2, target_normalized)
        if self.non_stochastic_version:
            kl_loss = torch.Tensor([0.0]).to(target_normalized.device)
            net_loss = recons_loss
        else:
            kl_loss = self.get_kl_divergence_loss(td_data)
            net_loss = recons_loss + self.get_kl_weight() * kl_loss

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
        return output


if __name__ == '__main__':
    import numpy as np
    import torch

    from denoisplit.configs.deepencoder_lvae_config import get_config

    config = get_config()
    data_mean = torch.Tensor([0]).reshape(1, 1, 1, 1)
    data_std = torch.Tensor([1]).reshape(1, 1, 1, 1)
    model = LVAEWithDeepEncoder(data_mean, data_std, config)
    mc = 1 if config.data.multiscale_lowres_count is None else config.data.multiscale_lowres_count + 1
    inp = torch.rand((2, mc, config.data.image_size, config.data.image_size))
    out1, out2, td_data = model(inp)
    print(out1.shape, out2.shape)

    # print(td_data)
    # decoder invariance.
    bu_values_l1 = []
    for i in range(1, len(config.model.z_dims) + 1):
        isz = config.data.image_size
        z = config.model.encoder.n_filters
        pow = 2**(i)
        bu_values_l1.append(torch.rand(2, z // 2, isz // pow, isz // pow))

    out_l1_1x, _ = model.topdown_pass(
        bu_values_l1,
        top_down_layers=model.top_down_layers_l1,
        final_top_down_layer=model.final_top_down_l1,
    )

    out_l1_10x, _ = model.topdown_pass(
        [10 * x for x in bu_values_l1],
        top_down_layers=model.top_down_layers_l1,
        final_top_down_layer=model.final_top_down_l1,
    )

    max_diff = torch.abs(out_l1_1x * 10 - out_l1_10x).max().item()
    assert max_diff < 1e-5
    out_l1_1x, _ = model.likelihood_l1.get_mean_lv(out_l1_1x)
    out_l1_10x, _ = model.likelihood_l1.get_mean_lv(out_l1_10x)
    max_diff = torch.abs(out_l1_1x * 10 - out_l1_10x).max().item()
    assert max_diff < 1e-5
    # out_l1_1x = model.top_down_layers_l1[0](None, bu_value=bu_values_l1[0], inference_mode=True,use_mode=True)
    # out_l1_10x = model.top_down_layers_l1[0](None, bu_value=10*bu_values_l1[0], inference_mode=True,use_mode=True)
    # inp, target, alpha_val, ch1_idx, ch2_idx
    batch = (torch.rand((16, mc, config.data.image_size, config.data.image_size)),
             torch.rand((16, 2, config.data.image_size, config.data.image_size)),
             torch.Tensor(np.random.randint(20, size=16)), torch.Tensor(np.random.randint(1000),
                                                                        np.random.randint(1000)))
    model.training_step(batch, 0)
    model.validation_step(batch, 0)

    print('mar')
