import os
from copy import deepcopy

import torch

import ml_collections
from denoisplit.config_utils import load_config
from denoisplit.nets.lvae import LadderVAE


class SplitterDenoiser(LadderVAE):
    """
    It denoises the splitted output. This is the second step in the pipeline of split=>denoise.
    We have 2 options for the denoise portion. 
    1. Do a unsupervised denoising. 
    2. Do a supervised denoising. This might even be useful to remove artefacts caused by the first model. 
    """

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        new_config = deepcopy(ml_collections.ConfigDict(config))
        with new_config.unlocked():
            new_config.data.color_ch = 2

        super().__init__(data_mean, data_std, new_config, use_uncond_mode_at, target_ch)

        self._splitter = self.load_splitter(config.model.pre_trained_ckpt_fpath_splitter)

    def load_data_mean_std(self, checkpoint):
        # TODO: save the mean and std in the checkpoint.
        data_mean = self.data_mean
        data_std = self.data_std
        return data_mean, data_std

    def load_splitter(self, pre_trained_ckpt_fpath):
        checkpoint = torch.load(pre_trained_ckpt_fpath)
        config_fpath = os.path.join(os.path.dirname(pre_trained_ckpt_fpath), 'config.pkl')
        config = load_config(config_fpath)
        data_mean, data_std = self.load_data_mean_std(checkpoint)
        model = LadderVAE(data_mean, data_std, config)
        _ = model.load_state_dict(checkpoint['state_dict'], strict=True)
        print('Loaded model from ckpt dir', pre_trained_ckpt_fpath, f' at epoch:{checkpoint["epoch"]}')

        for param in model.parameters():
            param.requires_grad = False
        return model

    def forward(self, x):
        x = self.get_splitted_output(x)
        return super().forward(x)

    def get_splitted_output(self, x):
        out, _ = self._splitter(x)
        return self._splitter.likelihood.distr_params(out)['mean']


if __name__ == '__main__':
    import numpy as np
    import torch

    from denoisplit.configs.splitter_denoiser_config import get_config

    config = get_config()
    data_mean = {'input': np.array([0]).reshape(1, 1, 1, 1), 'target': np.array([0, 0]).reshape(1, 2, 1, 1)}
    data_std = {'input': np.array([1]).reshape(1, 1, 1, 1), 'target': np.array([1, 1]).reshape(1, 2, 1, 1)}
    model = SplitterDenoiser(data_mean, data_std, config)
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
