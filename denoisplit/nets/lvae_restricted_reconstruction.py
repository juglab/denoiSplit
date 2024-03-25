import numpy as np

from denoisplit.core.loss_type import LossType
from denoisplit.loss.restricted_reconstruction_loss import RestrictedReconstruction
from denoisplit.nets.lvae import LadderVAE


class LadderVAERestrictedReconstruction(LadderVAE):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2, val_idx_manager=None):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at, target_ch, val_idx_manager=val_idx_manager)
        self.automatic_optimization = False
        assert self.loss_type == LossType.ElboRestrictedReconstruction
        self.mixed_rec_w = config.loss.mixed_rec_weight
        self.split_w = config.loss.get('split_weight', 1.0)
        self._switch_to_nonorthogonal_epoch = config.loss.get('switch_to_nonorthogonal_epoch', 100000)

        # note that split_s is directly multipled with the loss and not with the gradient.
        self.grad_setter = RestrictedReconstruction(1, self.mixed_rec_w)
        self._nonorthogonal_epoch_enabled = False

    def training_step(self, batch, batch_idx, enable_logging=True):
        if self.current_epoch == 0 and batch_idx == 0:
            self.log('val_psnr', 1.0, on_epoch=True)

        if self.current_epoch == self._switch_to_nonorthogonal_epoch and self._nonorthogonal_epoch_enabled == False:
            self.grad_setter.enable_nonorthogonal()
            self._nonorthogonal_epoch_enabled = True

        x, target = batch[:2]
        x_normalized = self.normalize_input(x)
        assert self.reconstruction_mode != True
        target_normalized = self.normalize_target(target)
        mask = ~((target == 0).reshape(len(target), -1).all(dim=1))
        out, td_data = self.forward(x_normalized)
        assert self.loss_type == LossType.ElboRestrictedReconstruction
        pred_x_normalized, _ = self.get_mixed_prediction(out, None, self.data_mean, self.data_std)
        optim = self.optimizers()
        optim.zero_grad()
        split_loss = self.grad_setter.loss_fn(target_normalized[mask], out[mask])
        self.manual_backward(self.split_w * split_loss, retain_graph=True)
        # add input reconstruction loss compoenent to the gradient.
        loss_dict = self.grad_setter.update_gradients(list(self.named_parameters()), x_normalized,
                                                      target_normalized[mask], out[mask], pred_x_normalized,
                                                      self.current_epoch)
        optim.step()
        assert self.non_stochastic_version == True
        if enable_logging:
            training_loss = self.split_w * split_loss + self.mixed_rec_w * loss_dict['input_reconstruction_loss']
            self.log('training_loss', training_loss, on_epoch=True)
            self.log('reconstruction_loss', split_loss, on_epoch=True)
            self.log('input_reconstruction_loss', loss_dict['input_reconstruction_loss'], on_epoch=True)
            for key in loss_dict['log']:
                self.log(key, loss_dict['log'][key], on_epoch=True)

    def on_validation_epoch_end(self):
        psnr_arr = []
        for i in range(len(self.channels_psnr)):
            psnr = self.channels_psnr[i].get()
            psnr_arr.append(psnr.cpu().numpy())
            self.channels_psnr[i].reset()

        psnr = np.mean(psnr_arr)
        self.log('val_psnr', psnr, on_epoch=True)

        sch1 = self.lr_schedulers()
        sch1.step(psnr)

        if self._dump_kth_frame_prediction is not None:
            if self.current_epoch == 0 or self.current_epoch % self._dump_epoch_interval == 0:
                self._val_frame_creator.dump(self.current_epoch)
                self._val_frame_creator.reset()
            if self.current_epoch == 1:
                self._val_frame_creator.dump_target()

        if self.mixed_rec_w_step:
            self.mixed_rec_w = max(self.mixed_rec_w - self.mixed_rec_w_step, 0.0)
            self.log('mixed_rec_w', self.mixed_rec_w, on_epoch=True)


if __name__ == '__main__':
    import numpy as np
    import torch

    from denoisplit.configs.biosr_sparsely_supervised_config import get_config
    config = get_config()
    # config.loss.critic_loss_weight = 0.0
    data_mean = torch.Tensor([0]).reshape(1, 1, 1, 1)
    data_std = torch.Tensor([1]).reshape(1, 1, 1, 1)
    model = LadderVAERestrictedReconstruction({
        'input': data_mean,
        'target': data_mean.repeat(1, 2, 1, 1)
    }, {
        'input': data_std,
        'target': data_std.repeat(1, 2, 1, 1)
    }, config)
    model.configure_optimizers()
    mc = 1 if config.data.multiscale_lowres_count is None else config.data.multiscale_lowres_count
    inp = torch.rand((2, mc, config.data.image_size, config.data.image_size))
    out, td_data = model(inp)
    batch = (
        torch.rand((16, mc, config.data.image_size, config.data.image_size)),
        torch.rand((16, 2, config.data.image_size, config.data.image_size)),
    )
    batch[1][::2] = 0 * batch[1][::2]

    model.validation_step(batch, 0)
    model.training_step(batch, 0)

    ll = torch.ones((12, 2, 32, 32))
    ll_new = model._get_weighted_likelihood(ll)
    print(ll_new[:, 0].mean(), ll_new[:, 0].std())
    print(ll_new[:, 1].mean(), ll_new[:, 1].std())
    print('mar')
