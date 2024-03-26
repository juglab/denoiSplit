import torch

from denoisplit.nets.lvae import LadderVAE


class LadderVAETranslation(LadderVAE):
    """
    It is a image translation network. the benefit is sampling.
    """

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[]):
        # since input is the target, we don't need to normalize it at all.
        super().__init__(data_mean, data_std, config, use_uncond_mode_at=use_uncond_mode_at, target_ch=1)
        assert config.data.target_separate_normalization == True
        assert config.data.normalized_input == False
        self._tar_idx = config.data.tar_idx
        self._inp_idx = config.data.inp_idx
        
        # NOTE: ideally, this should be done on the dataset side. 
        self.data_mean['input'] = self.data_mean['target'][:,self._inp_idx:self._inp_idx + 1].clone()
        self.data_std['input'] = self.data_std['target'][:,self._inp_idx: self._inp_idx + 1].clone()

        self.data_mean['target'] = self.data_mean['target'][:,self._tar_idx:self._tar_idx + 1].clone()
        self.data_std['target'] = self.data_std['target'][:,self._tar_idx: self._tar_idx + 1].clone()
        
        # if self.denoise_channel == 'all':
        #     msg = 'For target, we expect it to be unnormalized. For such reasons, we expect same normalization for input and target.'
        #     assert len(self.data_mean['target'].squeeze()) == 2, msg
        #     assert self.data_mean['input'].squeeze() == self.data_mean['target'].squeeze()[:1], msg
        #     assert self.data_mean['input'].squeeze() == self.data_mean['target'].squeeze()[1:], msg

        #     assert len(self.data_std['target'].squeeze()) == 2, msg
        #     assert self.data_std['input'].squeeze() == self.data_std['target'].squeeze()[:1], msg
        #     assert self.data_std['input'].squeeze() == self.data_std['target'].squeeze()[1:], msg
        #     self.data_mean['target'] = self.data_mean['target'][:, :1]
        #     self.data_std['target'] = self.data_std['target'][:, :1]
        # elif self.denoise_channel == 'input':
        #     self.data_mean['target'] = self.data_mean['input']
        #     self.data_std['target'] = self.data_std['input']
        # elif self.denoise_channel == 'Ch1':
        #     self.data_mean['target'] = self.data_mean['target'][:, :1]
        #     self.data_std['target'] = self.data_std['target'][:, :1]
        #     self.data_mean['input'] = self.data_mean['target']
        #     self.data_std['input'] = self.data_std['target']
        # elif self.denoise_channel == 'Ch2':
        #     self.data_mean['target'] = self.data_mean['target'][:, 1:]
        #     self.data_std['target'] = self.data_std['target'][:, 1:]
        #     self.data_mean['input'] = self.data_mean['target']
        #     self.data_std['input'] = self.data_std['target']

    # def get_new_input_target(self, batch):
    #     x, target = batch[:2]
    #     if self.denoise_channel == 'input':
    #         assert x.shape[1] == 1
    #         new_target = x.clone()
    #         # Input is normalized, but target is not. So we need to un-normalize it.
    #         new_target = new_target * self.data_std['input'] + self.data_mean['input']
    #     elif self.denoise_channel == 'Ch1':
    #         new_target = target[:, :1]
    #         # Input is normalized, but target is not. So we need to normalize it.
    #         x = self.normalize_target(new_target)

    #     elif self.denoise_channel == 'Ch2':
    #         new_target = target[:, 1:]
    #         # Input is normalized, but target is not. So we need to normalize it.
    #         x = self.normalize_target(new_target)
    #     elif self.denoise_channel == 'all':
    #         assert x.shape[1] == 1
    #         x = x * self.data_std['input'] + self.data_mean['input']
    #         new_target = torch.cat([x, target[:, :1], target[:, 1:]], dim=0)
    #         x = self.normalize_target(new_target)
    #     return x, new_target

    # def training_step(self, batch, batch_idx, enable_logging=True):
    #     x, target = batch[:2]
    #     x = target[:, self._inp_idx:self._inp_idx + 1].clone()
    #     target = target[:, self._tar_idx:self._tar_idx + 1].clone()
    #     batch = (x, target, *batch[2:])
    #     return super().training_step(batch, batch_idx, enable_logging)

    # def validation_step(self, batch, batch_idx):
    #     self.set_params_to_same_device_as(batch[0])
    #     x, target = batch[:2]
    #     target = target[:, self._tar_idx:self._tar_idx + 1].clone()
    #     batch = (x, target, *batch[2:])
    #     return super().validation_step(batch, batch_idx)


if __name__ == '__main__':
    import numpy as np
    import torch

    from denoisplit.configs.hdn_denoiser_config import get_config

    config = get_config()
    data_mean = {'input': np.array([0]).reshape(1, 1, 1, 1), 'target': np.array([0, 0]).reshape(1, 2, 1, 1)}
    data_std = {'input': np.array([1]).reshape(1, 1, 1, 1), 'target': np.array([1, 1]).reshape(1, 2, 1, 1)}
    import pdb
    pdb.set_trace()
    model = LadderVAEDenoiser(data_mean, data_std, config)
    mc = 1 if config.data.multiscale_lowres_count is None else config.data.multiscale_lowres_count + 1
    inp = torch.rand((2, mc, config.data.image_size, config.data.image_size))
    out, td_data = model(inp)
    print(out.shape)
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
