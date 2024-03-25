from denoisplit.configs.default_config import get_default_config
from denoisplit.core.data_type import DataType
from denoisplit.core.loss_type import LossType
from denoisplit.core.model_type import ModelType
from denoisplit.core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.data_type = DataType.Places365
    data.img_dsample = 2
    data.image_size = 128 // data.img_dsample
    data.label1 = 'ice_skating_rink-outdoor'
    data.label2 = 'waiting_room'
    data.sampler_type = SamplerType.RandomSampler
    data.return_img_labels = False

    loss = config.loss
    loss.loss_type = LossType.Elbo
    loss.kl_weight = 0.01
    loss.kl_annealing = False
    loss.kl_annealtime = 10
    loss.kl_start = -1
    loss.kl_min = 1e-7
    loss.free_bits = 0.0

    model = config.model
    model.model_type = ModelType.LadderVae
    model.z_dims = [128, 128, 128]
    model.encoder.blocks_per_layer = 3
    model.decoder.blocks_per_layer = 3
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

    training = config.training
    training.lr = 0.001
    training.lr_scheduler_patience = 30
    training.max_epochs = 1000
    training.batch_size = 16
    training.num_workers = 4
    return config
