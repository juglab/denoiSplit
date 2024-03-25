from denoisplit.configs.default_config import get_default_config
from denoisplit.core.data_type import DataType
from denoisplit.core.loss_type import LossType
from denoisplit.core.model_type import ModelType
from denoisplit.core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 64
    data.data_type = DataType.SemiSupBloodVesselsEMBL
    data.mix_fpath = ''  #THG-SJS42_0-1000_FITC_221116-1.tif'
    data.ch1_fpath = ''  #FITC_C1-SJS42_0-1000_FITC_221116-1.tif'
    data.mix_fpath_list = [
        'THG_MS29_z0_403um_sl4_bin10_z03_fr3_p9_lz290_px512_XYn119n152_AOFull_FITC_00002.tif',
        'THG_MS29_z0_905um_sl4_bin10_z03_fr3_p28_lz250_px512_XYn119n152_AOFull_FITC_00001.tif',
        'THG_MS29_z0_905um_sl4_bin10_z03_fr3_p33_lz250_px512_XYn119n152_AOFull_FITC_00001.tif'
    ]
    data.ch1_fpath_list = [x.replace('THG_', 'FITC_') for x in data.mix_fpath_list]

    # data.ignore_frames = [list(range(7)) + list(range(249, 260))]
    data.sampler_type = SamplerType.DefaultSampler
    data.threshold = 0.02
    data.deterministic_grid = False
    data.normalized_input = True
    data.clip_percentile = 0.995
    # If this is set to true, then one mean and stdev is used for both channels. Otherwise, two different
    # meean and stdev are used.
    data.use_one_mu_std = False
    # if this is set to True, then for each image, you normalize using it's mean and std.
    # data.use_per_image_mu_std = True
    data.train_aug_rotate = False
    data.randomized_channels = False
    data.multiscale_lowres_count = 3
    data.padding_mode = 'reflect'
    data.padding_value = None
    # If this is set to True, then target channels will be normalized from their separate mean.
    # otherwise, target will be normalized just the same way as the input, which is determined by use_one_mu_std
    data.target_separate_normalization = True

    loss = config.loss
    loss.loss_type = LossType.ElboSemiSupMixedReconstruction
    loss.mixed_rec_weight = 1
    loss.exclusion_loss_weight = 0.1

    loss.kl_weight = 1
    loss.kl_annealing = False
    loss.kl_annealtime = 10
    loss.kl_start = -1
    loss.kl_min = 1e-7
    loss.free_bits = 0.0

    model = config.model
    model.model_type = ModelType.LadderVaeSemiSupervised
    model.z_dims = [128, 128, 128, 128, 128, 128]

    model.encoder.blocks_per_layer = 1
    model.encoder.n_filters = 64
    model.encoder.dropout = 0.1
    model.encoder.res_block_kernel = 3
    model.encoder.res_block_skip_padding = False

    model.decoder.blocks_per_layer = 1
    model.decoder.n_filters = 64
    model.decoder.dropout = 0.1
    model.decoder.res_block_kernel = 3
    model.decoder.res_block_skip_padding = False
    #True

    model.skip_nboundary_pixels_from_loss = None
    model.nonlin = 'elu'
    model.merge_type = 'residual'
    model.batchnorm = True
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
    model.monitor = 'val_psnr'  # {'val_loss','val_psnr'}
    model.non_stochastic_version = False

    training = config.training
    training.lr = 0.001
    training.lr_scheduler_patience = 30
    training.max_epochs = 400
    training.batch_size = 32
    training.num_workers = 4
    training.val_repeat_factor = None
    training.train_repeat_factor = None
    training.val_fraction = 0.1
    training.test_fraction = 0.1
    training.earlystop_patience = 200
    training.precision = 16

    return config
