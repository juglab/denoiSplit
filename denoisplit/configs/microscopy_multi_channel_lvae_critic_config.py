"""
Configuration file for the VAE model with critic
"""
import ml_collections
from denoisplit.configs.default_config import get_default_config
from denoisplit.core.data_type import DataType
from denoisplit.core.loss_type import LossType
from denoisplit.core.model_type import ModelType
from denoisplit.core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 256
    data.data_type = DataType.OptiMEM100_014
    data.channel_1 = 0
    data.channel_2 = 2
    data.sampler_type = SamplerType.DefaultSampler
    data.threshold = 0.02
    data.deterministic_grid = True
    data.normalized_input = True
    # If this is set to true, then one mean and stdev is used for both channels. Otherwise, two different
    # meean and stdev are used.
    data.use_one_mu_std = True
    data.train_aug_rotate = True

    loss = config.loss
    loss.loss_type = LossType.ElboWithCritic
    loss.kl_weight = 0.005
    loss.kl_annealing = False
    loss.kl_annealtime = 10
    loss.kl_start = -1
    loss.kl_min = 1e-7
    loss.free_bits = 0.0
    loss.critic_loss_weight = 0.005

    model = config.model
    model.model_type = ModelType.LadderVAECritic
    model.z_dims = [128, 128, 128]
    model.encoder.blocks_per_layer = 5
    model.decoder.blocks_per_layer = 5
    model.nonlin = 'elu'
    model.merge_type = 'residual'
    model.batchnorm = True
    model.stochastic_skip = True
    model.n_filters = 64
    model.dropout = 0.2
    model.learn_top_prior = True
    model.img_shape = None
    model.res_block_type = 'bacdbacd'
    model.gated = True
    model.no_initial_downscaling = True
    model.analytical_kl = True
    model.mode_pred = False
    model.var_clip_max = 8
    # Discriminator params
    model.critic = ml_collections.ConfigDict()
    model.critic.ndf = 64
    model.critic.netD = 'n_layers'
    model.critic.layers_D = 2
    model.critic.norm = 'none'

    training = config.training
    training.lr = 0.001
    training.lr_scheduler_patience = 15
    training.max_epochs = 200
    training.batch_size = 4
    training.num_workers = 4
    training.val_repeat_factor = None
    training.train_repeat_factor = None
    training.val_fraction = 0.2
    training.earlystop_patience = 100
    training.precision = 16
    return config
