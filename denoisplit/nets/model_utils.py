import glob
import os
import pickle

import pytorch_lightning as pl
import torch
import torch.nn as nn

from denoisplit.config_utils import get_updated_config
from denoisplit.core.data_type import DataType
from denoisplit.core.loss_type import LossType
from denoisplit.core.model_type import ModelType
from denoisplit.nets.brave_net import BraveNetPL
from denoisplit.nets.denoiser_splitter import DenoiserSplitter
from denoisplit.nets.lvae import LadderVAE
from denoisplit.nets.lvae_bleedthrough import LadderVAEWithMixedRecons
from denoisplit.nets.lvae_deepencoder import LVAEWithDeepEncoder
from denoisplit.nets.lvae_denoiser import LadderVAEDenoiser
from denoisplit.nets.lvae_multidset_multi_input_branches import LadderVaeMultiDatasetMultiBranch
from denoisplit.nets.lvae_multidset_multi_optim import LadderVaeMultiDatasetMultiOptim
from denoisplit.nets.lvae_multiple_encoder_single_opt import LadderVAEMulEncoder1Optim
from denoisplit.nets.lvae_multiple_encoders import LadderVAEMultipleEncoders
from denoisplit.nets.lvae_multires_target import LadderVAEMultiTarget
from denoisplit.nets.lvae_restricted_reconstruction import LadderVAERestrictedReconstruction
from denoisplit.nets.lvae_semi_supervised import LadderVAESemiSupervised
from denoisplit.nets.lvae_twindecoder import LadderVAETwinDecoder
from denoisplit.nets.lvae_twodset import LadderVaeTwoDset
from denoisplit.nets.lvae_twodset_finetuning import LadderVaeTwoDsetFinetuning
from denoisplit.nets.lvae_twodset_restrictedrecons import LadderVaeTwoDsetRestrictedRecons
from denoisplit.nets.lvae_with_critic import LadderVAECritic
from denoisplit.nets.lvae_with_stitch import LadderVAEwithStitching
from denoisplit.nets.lvae_with_stitch_2stage import LadderVAEwithStitching2Stage
from denoisplit.nets.splitter_denoiser import SplitterDenoiser
from denoisplit.nets.unet import UNet


def create_model(config, data_mean, data_std, val_idx_manager=None):
    if config.model.model_type == ModelType.LadderVae:
        if 'num_targets' in config.model:
            target_ch = config.model.num_targets
        else:
            target_ch = config.data.get('num_channels', 2)

        model = LadderVAE(data_mean, data_std, config, target_ch=target_ch, val_idx_manager=val_idx_manager)
    elif config.model.model_type == ModelType.LadderVaeTwinDecoder:
        model = LadderVAETwinDecoder(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVAECritic:
        model = LadderVAECritic(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeSepEncoder:
        model = LadderVAEMultipleEncoders(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVAEMultiTarget:
        model = LadderVAEMultiTarget(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeSepEncoderSingleOptim:
        model = LadderVAEMulEncoder1Optim(data_mean, data_std, config)
    elif config.model.model_type == ModelType.UNet:
        model = UNet(data_mean, data_std, config)
    elif config.model.model_type == ModelType.BraveNet:
        model = BraveNetPL(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeStitch:
        model = LadderVAEwithStitching(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeMixedRecons:
        model = LadderVAEWithMixedRecons(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeSemiSupervised:
        model = LadderVAESemiSupervised(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeStitch2Stage:
        model = LadderVAEwithStitching2Stage(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeTwoDataSet:
        model = LadderVaeTwoDset(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeTwoDatasetMultiBranch:
        model = LadderVaeMultiDatasetMultiBranch(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVaeTwoDatasetMultiOptim:
        model = LadderVaeMultiDatasetMultiOptim(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LVaeDeepEncoderIntensityAug:
        model = LVAEWithDeepEncoder(data_mean, data_std, config)
    elif config.model.model_type == ModelType.Denoiser:
        model = LadderVAEDenoiser(data_mean, data_std, config)
    elif config.model.model_type == ModelType.DenoiserSplitter:
        model = DenoiserSplitter(data_mean, data_std, config)
    elif config.model.model_type == ModelType.SplitterDenoiser:
        model = SplitterDenoiser(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVAERestrictedReconstruction:
        model = LadderVAERestrictedReconstruction(data_mean, data_std, config, val_idx_manager=val_idx_manager)
    elif config.model.model_type == ModelType.LadderVAETwoDataSetRestRecon:
        model = LadderVaeTwoDsetRestrictedRecons(data_mean, data_std, config)
    elif config.model.model_type == ModelType.LadderVAETwoDataSetFinetuning:
        model = LadderVaeTwoDsetFinetuning(data_mean, data_std, config)
    else:
        raise Exception('Invalid model type:', config.model.model_type)

    if config.model.get('pretrained_weights_path', None):
        ckpt_fpath = config.model.pretrained_weights_path
        checkpoint = torch.load(ckpt_fpath)
        skip_likelihood = config.model.get('pretrained_weights_skip_likelihood', False)
        if skip_likelihood:
            checkpoint['state_dict'].pop('likelihood.parameter_net.weight')
            checkpoint['state_dict'].pop('likelihood.parameter_net.bias')

        _ = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print('Loaded model from ckpt dir', ckpt_fpath, f' at epoch:{checkpoint["epoch"]}')

    return model


def get_best_checkpoint(ckpt_dir):
    output = []
    for filename in glob.glob(ckpt_dir + "/*_best.ckpt"):
        output.append(filename)
    assert len(output) == 1, '\n'.join(output)
    return output[0]


def load_model_checkpoint(ckpt_dir: str,
                          data_mean: float,
                          data_std: float,
                          config=None,
                          model=None) -> pl.LightningModule:
    """
    It loads the model from the checkpoint directory
    """
    import ml_collections  # Needed due to loading in pickle
    if model is None:
        # load config, if the config is not provided
        if config is None:
            with open(os.path.join(ckpt_dir, 'config.pkl'), 'rb') as f:
                config = pickle.load(f)

        config = get_updated_config(config)
        model = create_model(config, data_mean, data_std)
    ckpt_fpath = get_best_checkpoint(ckpt_dir)
    checkpoint = torch.load(ckpt_fpath)
    _ = model.load_state_dict(checkpoint['state_dict'])
    print('Loaded model from ckpt dir', ckpt_dir, f' at epoch:{checkpoint["epoch"]}')
    return model
