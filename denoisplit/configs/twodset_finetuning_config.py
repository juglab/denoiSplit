from tkinter.tix import Tree

import numpy as np

import ml_collections
from denoisplit.configs.default_config import get_default_config
from denoisplit.core.data_type import DataType
from denoisplit.core.loss_type import LossType
from denoisplit.core.model_type import ModelType
from denoisplit.core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 128
    data.data_type = DataType.TwoDset
    data.channel_1 = None
    data.channel_2 = None
    data.ch1_fname = ''
    data.ch2_fname = ''
    # Specific to TwoDset
    data.dset0 = ml_collections.ConfigDict()
    data.dset0.data_type = DataType.BioSR_MRC
    data.dset0.ch1_fname = 'ER/GT_all.mrc'
    data.dset0.ch2_fname = 'Microtubules/GT_all.mrc'
    data.dset0.synthetic_gaussian_scale = 6675
    data.dset0.poisson_noise_factor = 1000
    data.dset0.enable_gaussian_noise = True

    data.dset1 = ml_collections.ConfigDict()
    data.dset1.data_type = DataType.BioSR_MRC
    data.dset1.ch1_fname = 'ER/GT_all.mrc'
    data.dset1.ch2_fname = 'Microtubules/GT_all.mrc'
    data.dset1.synthetic_gaussian_scale = 4450
    data.dset1.poisson_noise_factor = 1000
    data.dset1.enable_gaussian_noise = True
    data.subdset_types_probab = [0.5, 0.5]
    #############################

    data.poisson_noise_factor = 1000

    data.enable_gaussian_noise = True
    data.trainig_datausage_fraction = 1.0
    # data.validtarget_random_fraction = 1.0
    # data.training_validtarget_fraction = 0.2
    config.data.synthetic_gaussian_scale = 4450
    # if True, then input has 'identical' noise as the target. Otherwise, noise of input is independently sampled.
    config.data.input_has_dependant_noise = True

    data.sampler_type = SamplerType.DefaultSampler
    data.threshold = 0.02
    # data.grid_size = 1
    data.deterministic_grid = False
    data.normalized_input = True
    data.clip_percentile = 1

    data.channelwise_quantile = False
    # If this is set to true, then one mean and stdev is used for both channels. Otherwise, two different
    # meean and stdev are used.
    data.use_one_mu_std = True
    data.train_aug_rotate = False
    data.randomized_channels = False
    data.multiscale_lowres_count = None
    data.padding_mode = 'reflect'
    data.padding_value = None
    # If this is set to True, then target channels will be normalized from their separate mean.
    # otherwise, target will be normalized just the same way as the input, which is determined by use_one_mu_std
    data.target_separate_normalization = True
    data.input_is_sum = False
    loss = config.loss
    # loss.loss_type = LossType.Elbo
    loss.loss_type = LossType.ElboRestrictedReconstruction
    # this is not uSplit.
    loss.kl_loss_formulation = ''

    loss.mixed_rec_weight = 1.0
    loss.split_weight = 1.0
    loss.kl_weight = 1.0
    # loss.reconstruction_weight = 1.0
    loss.kl_annealing = False
    loss.kl_annealtime = 10
    loss.kl_start = -1
    loss.kl_min = 1e-7
    loss.free_bits = 1.0

    model = config.model
    model.model_type = ModelType.LadderVAETwoDataSetFinetuning
    model.z_dims = [128, 128, 128, 128]

    model.encoder.batchnorm = True
    model.encoder.blocks_per_layer = 1
    model.encoder.n_filters = 64
    model.encoder.dropout = 0.1
    model.encoder.res_block_kernel = 3
    model.encoder.res_block_skip_padding = False

    model.decoder.batchnorm = True
    model.decoder.blocks_per_layer = 1
    model.decoder.n_filters = 64
    model.decoder.dropout = 0.1
    model.decoder.res_block_kernel = 3
    model.decoder.res_block_skip_padding = False

    #False
    config.model.decoder.conv2d_bias = True

    model.skip_nboundary_pixels_from_loss = None
    model.nonlin = 'elu'
    model.merge_type = 'residual'
    model.stochastic_skip = True
    model.learn_top_prior = True
    model.img_shape = None
    model.res_block_type = 'bacdbacd'

    model.gated = True
    model.no_initial_downscaling = True
    model.analytical_kl = False
    model.mode_pred = False
    model.var_clip_max = 20
    # predict_logvar takes one of the four values: [None,'global','channelwise','pixelwise']
    model.predict_logvar = 'pixelwise'
    model.logvar_lowerbound = -5  # -2.49 is log(1/12), from paper "Re-parametrizing VAE for stablity."
    model.multiscale_lowres_separate_branch = False
    model.multiscale_retain_spatial_dims = True
    model.monitor = 'val_loss'  # {'val_loss','val_psnr'}
    model.non_stochastic_version = False
    model.enable_noise_model = False
    model.noise_model_type = 'gmm'
    # model.noise_model_ch1_fpath = '/home/ashesh.ashesh/training/noise_model/2402/226/GMMNoiseModel_ER-GT_all.mrc__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
    # model.noise_model_ch2_fpath = '/home/ashesh.ashesh/training/noise_model/2402/206/GMMNoiseModel_CCPs-GT_all.mrc__6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'

    #################
    # this must be the input.
    model.finetuning_noise_model_ch1_fpath = '/group/jug/ashesh/training_pre_eccv/noise_model/2402/475/GMMNoiseModel_BioSR-ER_GT_all_Microtubules_GT_all_6_4_Clip0.0-1.0_Sig0.125_UpNone_Norm0_bootstrap.npz'
    model.finetuning_noise_model_ch2_fpath = ''
    model.finetuning_noise_model_type = 'gmm'
    model.pretrained_weights_path = '/home/ashesh.ashesh/training/disentangle/2403/D16-M3-S0-L0/2/BaselineVAECL_best.ckpt'
    ################

    training = config.training
    training.lr = 0.001
    training.lr_scheduler_patience = 1
    training.max_epochs = 10
    training.batch_size = 32
    training.num_workers = 4
    training.val_repeat_factor = None
    training.train_repeat_factor = None
    training.val_fraction = 0.1
    training.test_fraction = 0.1
    training.earlystop_patience = 2
    # training.precision = 16

    return config
