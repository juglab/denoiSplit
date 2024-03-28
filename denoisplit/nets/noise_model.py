import json
import os

import numpy as np
import torch
import torch.nn as nn

from denoisplit.core.model_type import ModelType
from denoisplit.nets.gmm_nnbased_noise_model import DeepGMMNoiseModel
from denoisplit.nets.gmm_noise_model import GaussianMixtureNoiseModel
from denoisplit.nets.hist_gmm_noise_model import HistGMMNoiseModel
from denoisplit.nets.hist_noise_model import HistNoiseModel


class DisentNoiseModel(nn.Module):

    def __init__(self, n1model, n2model):
        super().__init__()
        self.n1model = n1model
        self.n2model = n2model

    def likelihood(self, obs, signal):
        if obs.shape[1] == 1:
            assert signal.shape[1] == 1
            assert self.n2model is None
            return self.n1model.likelihood(obs, signal)

        ll1 = self.n1model.likelihood(obs[:, :1], signal[:, :1])
        ll2 = self.n2model.likelihood(obs[:, 1:], signal[:, 1:])
        return torch.cat([ll1, ll2], dim=1)


def last2path(fpath):
    return os.path.join(*fpath.split('/')[-2:])


def noise_model_config_sanity_check(noise_model_fpath, config, channel_key=None):
    config_fpath = os.path.join(os.path.dirname(noise_model_fpath), 'config.json')
    with open(config_fpath, 'r') as f:
        noise_model_config = json.load(f)
    # make sure that the amount of noise is consistent.
    if 'add_gaussian_noise_std' in noise_model_config:
        # data.enable_gaussian_noise = False
        # config.data.synthetic_gaussian_scale = 1000
        assert 'enable_gaussian_noise' in config.data
        assert config.data.enable_gaussian_noise == True, 'Gaussian noise is not enabled'

        assert 'synthetic_gaussian_scale' in config.data
        assert noise_model_config[
            'add_gaussian_noise_std'] == config.data.synthetic_gaussian_scale, f'{noise_model_config["add_gaussian_noise_std"]} != {config.data.synthetic_gaussian_scale}'

    cfg_poisson_noise_factor = config.data.get('poisson_noise_factor', -1)
    nm_poisson_noise_factor = noise_model_config.get('poisson_noise_factor', -1)
    assert cfg_poisson_noise_factor == nm_poisson_noise_factor, f'{nm_poisson_noise_factor} != {cfg_poisson_noise_factor}'

    if 'train_pure_noise_model' in noise_model_config and noise_model_config['train_pure_noise_model']:
        print('Pure noise model is being used now.')
        return
    # make sure that the same file is used for noise model and data.
    if channel_key is not None and channel_key in noise_model_config:
        fname = noise_model_config['fname']
        if '' in fname:
            fname.remove('')
        assert len(fname) == 1
        fname = fname[0]
        cur_data_fpath = os.path.join(config.datadir, config.data[channel_key])
        nm_data_fpath = os.path.join(noise_model_config['datadir'], fname)
        if cur_data_fpath != nm_data_fpath:
            print(f'Warning: {cur_data_fpath} != {nm_data_fpath}')
            assert last2path(cur_data_fpath) == last2path(nm_data_fpath), f'{cur_data_fpath} != {nm_data_fpath}'
        # assert cur_data_fpath == nm_data_fpath, f'{cur_data_fpath} != {nm_data_fpath}'
    else:
        print(f'Warning: channel_key is not found in noise model config: {channel_key}')


def get_noise_model(config):
    if config.model.noise_model_type == 'hist':
        print(f'Noise model Ch1: {config.model.noise_model_ch1_fpath}')
        hist1 = np.load(config.model.noise_model_ch1_fpath)
        nmodel1 = HistNoiseModel(hist1)
        nmodel2 = None
    elif config.model.noise_model_type == 'gmm':
        nmodel_fpath = config.model.noise_model_ch1_fpath
        print(f'Noise model Ch1: {nmodel_fpath}')
        nmodel1 = GaussianMixtureNoiseModel(params=np.load(nmodel_fpath))
        noise_model_config_sanity_check(nmodel_fpath, config, 'ch1_fname')
        nmodel2 = None
        if config.model.get('noise_model_learnable', False):
            nmodel1.make_learnable()
            if nmodel2 is not None:
                nmodel2.make_learnable()

        return DisentNoiseModel(nmodel1, nmodel2)
    return None
