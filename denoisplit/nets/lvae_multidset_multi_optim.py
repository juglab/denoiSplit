import torch.nn as nn
import torch.optim as optim

from denoisplit.core.loss_type import LossType
from denoisplit.nets.lvae_multidset_multi_input_branches import LadderVaeMultiDatasetMultiBranch


class IntensityMap(nn.Module):

    def __init__(self):
        super().__init__()
        self._net = nn.Sequential(
            nn.Conv2d(1, 64, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        return x + self._net(x)


class LadderVaeMultiDatasetMultiOptim(LadderVaeMultiDatasetMultiBranch):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at, target_ch)

        self.automatic_optimization = False
        self._donot_keep_separate_firstbottomup = config.model.get('only_optimize_interchannel_weights', False)
        if self._donot_keep_separate_firstbottomup is True:
            del self._first_bottom_up_subdset0
            self._first_bottom_up_subdset0 = self._first_bottom_up_subdset1

        learn_imap = config.model.get('learn_intensity_map', False)
        self._intensity_map_net = None
        if learn_imap:
            self._intensity_map_net = IntensityMap()
            self._first_bottom_up_subdset0 = nn.Sequential(self._intensity_map_net, self._first_bottom_up_subdset0)

        print(
            f'[{self.__class__.__name__}] OnlyOptimizeInterchannelWeights:{self._donot_keep_separate_firstbottomup} IMap:{learn_imap}'
        )

    def get_encoder_params(self):
        encoder_params = list(self._first_bottom_up_subdset1.parameters()) + list(self.bottom_up_layers.parameters())
        if self.lowres_first_bottom_ups is not None:
            encoder_params.append(self.lowres_first_bottom_ups.parameters())
        return encoder_params

    def get_decoder_params(self):
        decoder_params = list(self.top_down_layers.parameters()) + list(self.final_top_down.parameters()) + list(
            self.likelihood.parameters())
        return decoder_params

    def get_mixrecons_extra_params(self):
        if self._donot_keep_separate_firstbottomup:
            params = []
            assert self._interchannel_weights is not None, "There would be nothing to optimize for the second optimizer."
        else:
            params = list(self._first_bottom_up_subdset0.parameters())

        if self._intensity_map_net is not None:
            params += list(self._intensity_map_net.parameters())

        if self._interchannel_weights is not None:
            params = params + [self._interchannel_weights]

        return params

    def get_scheduler(self, optimizer):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    self.lr_scheduler_mode,
                                                    patience=self.lr_scheduler_patience,
                                                    factor=0.5,
                                                    min_lr=1e-12,
                                                    verbose=True)

    def configure_optimizers(self):

        encoder_params = self.get_encoder_params()
        decoder_params = self.get_decoder_params()
        # channel 1 params
        ch2_pathway = encoder_params + decoder_params
        optimizer0 = optim.Adamax(ch2_pathway, lr=self.lr, weight_decay=0)

        optimizer1 = optim.Adamax(self.get_mixrecons_extra_params(), lr=self.lr, weight_decay=0)

        scheduler0 = self.get_scheduler(optimizer0)
        scheduler1 = self.get_scheduler(optimizer1)

        return [optimizer0, optimizer1], [{
            'scheduler': scheduler,
            'monitor': self.lr_scheduler_monitor,
        } for scheduler in [scheduler0, scheduler1]]

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target, dset_idx, loss_idx = batch
        ch2_opt, mix_opt = self.optimizers()
        mask_ch2 = loss_idx == LossType.Elbo
        mask_mix = loss_idx == LossType.ElboMixedReconstruction
        assert mask_ch2.sum() + mask_mix.sum() == len(x)
        loss_dict = None

        if mask_ch2.sum() > 0:
            batch = (x[mask_ch2], target[mask_ch2], dset_idx[mask_ch2], loss_idx[mask_ch2])
            loss_dict = super().training_step(batch, batch_idx, enable_logging=enable_logging)
            if loss_dict is not None:
                ch2_opt.zero_grad()
                loss = loss_dict['kl_loss'] + loss_dict['reconstruction_loss']
                self.manual_backward(loss)
                ch2_opt.step()

        if mask_mix.sum() > 0:
            batch = (x[mask_mix], target[mask_mix], dset_idx[mask_mix], loss_idx[mask_mix])
            mix_loss_dict = super().training_step(batch, batch_idx, enable_logging=enable_logging)
            if loss_dict is not None:
                mix_opt.zero_grad()
                loss = mix_loss_dict['kl_loss'] + mix_loss_dict['mixed_loss']
                self.manual_backward(loss)
                mix_opt.step()

        if loss_dict is not None:
            self.log_dict({"loss": loss}, prog_bar=True)


if __name__ == '__main__':
    import torch

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
    model = LadderVaeMultiDatasetMultiOptim(data_mean, data_std, config)
    dset_idx = torch.Tensor([0, 1, 0, 1])
    loss_idx = torch.Tensor(
        [LossType.Elbo, LossType.ElboMixedReconstruction, LossType.Elbo, LossType.ElboMixedReconstruction])
    x = torch.rand((4, 1, 64, 64))
    target = torch.rand((4, 2, 64, 64))
    batch = (x, target, dset_idx, loss_idx)
    _ = model.forward(x, 2)
    model.training_step(batch, 0, enable_logging=True)
    model.validation_step(batch, 0)
