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
    # note that this is dependant on image_size.
    # data.std_background_arr = [500.0, 500.0]
    # data.channel_1 = 0
    # data.channel_2 = 1
    data.ch1_fname = 'Microtubules/GT_all.mrc'
    data.ch2_fname = 'ER/GT_all.mrc'

    # amounnt of data (supervised and unsupervised) which you want to use for training.
    data.trainig_datausage_fraction = 1
    # how much data will use the target.
    data.training_validtarget_fraction = 0.01
    # when creating a batch, what fraction of inputs should have target.
    data.validtarget_random_fraction = 0.5

    data.validation_datausage_fraction = 0.08
    data.return_index = True

    # data.validtarget_random_fraction_final = 1
    # data.validtarget_random_fraction_stepepoch = 0.005

    data.sampler_type = SamplerType.DefaultSampler
    data.deterministic_grid = True
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
    data.train_aug_rotate = True
    data.randomized_channels = False
    # if multiscale_lowres_count is 3, then there are two additional inputs other than the original input. input channel count is 3
    data.multiscale_lowres_count = None
    data.padding_mode = 'reflect'
    data.padding_value = None
    # If this is set to True, then target channels will be normalized from their separate mean.
    # otherwise, target will be normalized just the same way as the input, which is determined by use_one_mu_std
    data.target_separate_normalization = True

    loss = config.loss
    loss.loss_type = LossType.ElboRestrictedReconstruction
    # loss.D_epsilon = 0.1
    # loss.critic_loss_weight = 0.001
    loss.mixed_rec_weight = 100.0
    loss.split_weight = 0.0
    loss.switch_to_nonorthogonal_epoch = 100000
    # loss.mixed_rec_w_step = 0.01
    # loss.exclusion_loss_weight = 0.005

    loss.kl_weight = 1
    loss.kl_annealing = False
    loss.kl_annealtime = 10
    loss.kl_start = -1
    loss.kl_min = 1e-7
    loss.free_bits = 0.0

    # loss.ch1_recons_w = 1
    # loss.ch2_recons_w = 5

    model = config.model
    model.model_type = ModelType.LadderVAERestrictedReconstruction
    # model.classifier_fpath = '/home/ubuntu/ashesh/training/disentangle/texture_classifier.pth'
    # model.classifier_loss_weight = 0.01

    model.z_dims = [128, 128, 128, 128]
    model.tethered_to_input = False
    # model.tethered_learnable_scalar = True
    # model.D_num_blocks_per_layer = 1
    # model.D_num_hierarchy_levels = 1
    # model.D_input_downsampling_count = 2

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
    model.reconstruction_mode = False
    model.skip_bottomk_buvalues = 0

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
    model.predict_logvar = None  #'ch_invariant_pixelwise'
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
    training.lr = 0.001
    training.lr_scheduler_patience = int(30 / data.trainig_datausage_fraction if 'trainig_datausage_fraction' in
                                         data else 30)
    training.max_epochs = int(200 / data.trainig_datausage_fraction if 'trainig_datausage_fraction' in data else 200)
    training.batch_size = 16
    training.num_workers = 4
    training.val_repeat_factor = None
    training.train_repeat_factor = None
    training.val_fraction = 0.1
    training.test_fraction = 0.1
    training.dump_epoch_interval = 10
    training.dump_kth_frame_prediction = 0

    training.earlystop_patience = int(100 /
                                      data.trainig_datausage_fraction if 'trainig_datausage_fraction' in data else 100)
    training.precision = 32
    training.check_val_every_n_epoch = int(
        1 / (data.trainig_datausage_fraction)) if 'trainig_datausage_fraction' in data else None

    return config
