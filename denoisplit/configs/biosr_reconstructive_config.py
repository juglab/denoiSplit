from denoisplit.configs.default_config import get_default_config
from denoisplit.core.data_type import DataType
from denoisplit.core.loss_type import LossType
from denoisplit.core.model_type import ModelType
from denoisplit.core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 128
    data.grid_size = 1
    data.data_type = DataType.BioSR_MRC
    # data.channel_1 = 0
    # data.channel_2 = 1
    data.ch1_fname = 'Microtubules/GT_all.mrc'
    data.ch2_fname = 'ER/GT_all.mrc'

    # amounnt of data (supervised and unsupervised) which you want to use for training.
    data.trainig_datausage_fraction = 1
    data.training_validtarget_fraction = None
    # when creating a batch, what fraction of inputs should have target.
    data.validtarget_random_fraction = None
    # data.validtarget_random_fraction_final = 0.9
    # data.validtarget_random_fraction_stepepoch = 0.005

    data.sampler_type = SamplerType.DefaultSampler
    data.deterministic_grid = False
    data.normalized_input = True
    data.clip_percentile = 0.995
    data.background_quantile = 0.0
    # With background quantile, one is setting the avg background value to 0. With this, any negative values are also set to 0.
    # This, together with correct background_quantile should altogether get rid of the background. The issue here is that
    # the background noise is also a distribution. So, some amount of background noise will remain.
    data.clip_background_noise_to_zero = False

    # we will not subtract the mean of the dataset from every patch. We just want to subtract the background and normalize using std. This way, background will be very close to 0.
    # this will help in the all scaling related approaches where we want to multiply the frame with some factor and then add them. we will then effectively just do these scaling on the
    # foreground pixels and the background will anyways will remain very close to 0.
    data.skip_normalization_using_mean = False

    data.input_is_sum = False

    # If this is set to true, then one mean and stdev is used for both channels. Otherwise, two different
    # meean and stdev are used.
    data.use_one_mu_std = True
    data.train_aug_rotate = False
    data.randomized_channels = False
    # if multiscale_lowres_count is 3, then there are two additional inputs other than the original input. input channel count is 3
    data.multiscale_lowres_count = None
    data.padding_mode = 'reflect'
    data.padding_value = None
    # If this is set to True, then target channels will be normalized from their separate mean.
    # otherwise, target will be normalized just the same way as the input, which is determined by use_one_mu_std
    data.target_separate_normalization = False

    loss = config.loss
    loss.loss_type = LossType.Elbo
    loss.mixed_rec_weight = 1

    loss.kl_weight = 1
    loss.kl_annealing = False
    loss.kl_annealtime = 10
    loss.kl_start = -1
    loss.kl_min = 1e-7
    loss.free_bits = 0.0
    # loss.ch1_recons_w = 1
    # loss.ch2_recons_w = 5

    model = config.model
    model.model_type = ModelType.LadderVae
    model.z_dims = [128, 128, 128, 128]
    model.skip_bottomk_buvalues = 3

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

    model.decoder.multiscale_retain_spatial_dims = True
    model.decoder.conv2d_bias = True
    model.reconstruction_mode = True

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
    # predict_logvar takes one of the four values: [None,'global','channelwise','pixelwise', 'ch_invariant_pixelwise]
    model.predict_logvar = None
    model.logvar_lowerbound = -5  # -2.49 is log(1/12), from paper "Re-parametrizing VAE for stablity."
    model.multiscale_lowres_separate_branch = False
    model.multiscale_retain_spatial_dims = True
    model.monitor = 'val_psnr'  # {'val_loss','val_psnr'}
    model.non_stochastic_version = True
    model.enable_noise_model = False
    model.noise_model_ch1_fpath = None
    model.noise_model_ch1_fpath = None
    # model.pretrained_weights_path = '/home/ashesh.ashesh/training/disentangle/2311/D16-M3-S0-L0/11/BaselineVAECL_best.ckpt'

    training = config.training
    training.lr = 0.001 / 2
    training.lr_scheduler_patience = int(60 / data.trainig_datausage_fraction if 'trainig_datausage_fraction' in
                                         data else 60)
    training.max_epochs = int(400 / data.trainig_datausage_fraction if 'trainig_datausage_fraction' in data else 400)
    training.batch_size = 32
    training.num_workers = 2
    training.val_repeat_factor = None
    training.train_repeat_factor = None
    training.val_fraction = 0.1
    training.test_fraction = 0.1

    training.earlystop_patience = int(200 /
                                      data.trainig_datausage_fraction if 'trainig_datausage_fraction' in data else 200)
    training.precision = 16
    training.check_val_every_n_epoch = int(
        1 / (data.trainig_datausage_fraction)) if 'trainig_datausage_fraction' in data else None

    return config
