"""
It has 4 channels: Nucleus, Nucleus, Actin, Tubulin
It has 3 sets: Only CYAN, ONLY MAGENTA, MIXED.
It has 2 versions: denoised and raw data.
"""
import os
import numpy as np
# from nd2reader import ND2Reader
from denoisplit.core.data_split_type import DataSplitType, get_datasplit_tuples
from denoisplit.data_loader.pavia2_enums import Pavia2DataSetType, Pavia2DataSetChannels, Pavia2DataSetVersion


def load_nd2(fpaths):
    """
    Load .nd2 images.
    """
    images = []
    for fpath in fpaths:
        with ND2Reader(fpath) as img:
            # channels are the last dimension.
            img = np.concatenate([x[..., None] for x in img], axis=-1)
            images.append(img[None])
    # number of images is the first dimension.
    return np.concatenate(images, axis=0)


def get_mixed_fnames(version):
    if version == Pavia2DataSetVersion.RAW:
        return [
            'HaCaT005.nd2', 'HaCaT009.nd2', 'HaCaT013.nd2', 'HaCaT016.nd2', 'HaCaT019.nd2', 'HaCaT029.nd2',
            'HaCaT037.nd2', 'HaCaT041.nd2', 'HaCaT044.nd2', 'HaCaT051.nd2', 'HaCaT054.nd2', 'HaCaT059.nd2',
            'HaCaT066.nd2', 'HaCaT071.nd2', 'HaCaT006.nd2', 'HaCaT011.nd2', 'HaCaT014.nd2', 'HaCaT017.nd2',
            'HaCaT020.nd2', 'HaCaT031.nd2', 'HaCaT039.nd2', 'HaCaT042.nd2', 'HaCaT045.nd2', 'HaCaT052.nd2',
            'HaCaT056.nd2', 'HaCaT063.nd2', 'HaCaT067.nd2', 'HaCaT007.nd2', 'HaCaT012.nd2', 'HaCaT015.nd2',
            'HaCaT018.nd2', 'HaCaT027.nd2', 'HaCaT034.nd2', 'HaCaT040.nd2', 'HaCaT043.nd2', 'HaCaT046.nd2',
            'HaCaT053.nd2', 'HaCaT058.nd2', 'HaCaT065.nd2', 'HaCaT068.nd2'
        ]


def get_justcyan_fnames(version):
    if version == Pavia2DataSetVersion.RAW:
        return [
            'HaCaT023.nd2', 'HaCaT024.nd2', 'HaCaT026.nd2', 'HaCaT032.nd2', 'HaCaT033.nd2', 'HaCaT036.nd2',
            'HaCaT048.nd2', 'HaCaT049.nd2', 'HaCaT057.nd2', 'HaCaT060.nd2', 'HaCaT062.nd2'
        ]


def get_justmagenta_fnames(version):
    if version == Pavia2DataSetVersion.RAW:
        return [
            'HaCaT008.nd2', 'HaCaT021.nd2', 'HaCaT025.nd2', 'HaCaT030.nd2', 'HaCaT038.nd2', 'HaCaT050.nd2',
            'HaCaT061.nd2', 'HaCaT069.nd2', 'HaCaT010.nd2', 'HaCaT022.nd2', 'HaCaT028.nd2', 'HaCaT035.nd2',
            'HaCaT047.nd2', 'HaCaT055.nd2', 'HaCaT064.nd2', 'HaCaT070.nd2'
        ]


def version_dir(dset_version):
    if dset_version == Pavia2DataSetVersion.RAW:
        return "RAW_DATA"
    elif dset_version == Pavia2DataSetVersion.DD:
        return "DD"


def load_data(datadir, dset_type, dset_version=Pavia2DataSetVersion.RAW):
    print(f'Loading Data from', datadir, Pavia2DataSetType.name(dset_type), Pavia2DataSetVersion.name(dset_version))
    if dset_type == Pavia2DataSetType.JustCYAN:
        datadir = os.path.join(datadir, version_dir(dset_version), 'ONLY_CYAN')
        fnames = get_justcyan_fnames(dset_version)
    elif dset_type == Pavia2DataSetType.JustMAGENTA:
        datadir = os.path.join(datadir, version_dir(dset_version), 'ONLY_MAGENTA')
        fnames = get_justmagenta_fnames(dset_version)
    elif dset_type == Pavia2DataSetType.MIXED:
        datadir = os.path.join(datadir, version_dir(dset_version), 'MIXED')
        fnames = get_mixed_fnames(dset_version)

    fpaths = [os.path.join(datadir, x) for x in fnames]
    data = load_nd2(fpaths)
    return data


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    dset_type = data_config.dset_type
    data = load_data(datadir, dset_type)
    data = data[..., data_config.channel_idx_list]
    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(data))
    if datasplit_type == DataSplitType.All:
        data = data.astype(np.float32)
    elif datasplit_type == DataSplitType.Train:
        data = data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        data = data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        data = data[test_idx].astype(np.float32)
    else:
        raise Exception("invalid datasplit")

    return data


def get_train_val_data_vanilla(datadir,
                               data_config,
                               datasplit_type: DataSplitType,
                               val_fraction=None,
                               test_fraction=None):
    dset_type = Pavia2DataSetType.JustMAGENTA
    data = load_data(datadir, dset_type)
    data = data[..., [data_config.channel_1, data_config.channel_2]]
    data[..., 1] = data[..., 1] / data_config.channel_2_downscale_factor
    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(data))
    if datasplit_type == DataSplitType.All:
        data = data.astype(np.float32)
    elif datasplit_type == DataSplitType.Train:
        data = data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        data = data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        data = data[test_idx].astype(np.float32)
    else:
        raise Exception("invalid datasplit")

    return data
