import os

import numpy as np

# from czifile import imread as imread_czi
from denoisplit.core.custom_enum import Enum
from denoisplit.core.data_split_type import DataSplitType, get_datasplit_tuples


class SubDsetType(Enum):
    OnlyIba1 = 'Iba1'
    Iba1Ki64 = 'Iba1_Ki67'


def get_iba1_ki67_files():
    return [f'{i}.czi' for i in range(1, 31)]


def get_iba1_only_files():
    return [f'Iba1only_{i}.czi' for i in range(1, 16)]


def load_czi(fpaths):
    imgs = []
    for fpath in fpaths:
        img = imread_czi(fpath)
        assert img.shape[3] == 1
        img = np.swapaxes(img, 0, 3)
        # the first dimension of img stored in imgs will have dim of 1, where the contenation will happen
        imgs.append(img)
    return np.concatenate(imgs, axis=0)


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    dset_subtype = data_config.subdset_type

    if dset_subtype == SubDsetType.OnlyIba1:
        fnames = get_iba1_only_files()
    elif dset_subtype == SubDsetType.Iba1Ki64:
        fnames = get_iba1_ki67_files()

    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(fnames))
    if datasplit_type == DataSplitType.All:
        pass
    elif datasplit_type == DataSplitType.Train:
        print(train_idx)
        fnames = [fnames[i] for i in train_idx]
    elif datasplit_type == DataSplitType.Val:
        print(val_idx)
        fnames = [fnames[i] for i in val_idx]
    elif datasplit_type == DataSplitType.Test:
        print(test_idx)
        fnames = [fnames[i] for i in test_idx]
    else:
        raise Exception("invalid datasplit")

    fpaths = [os.path.join(datadir, dset_subtype, x) for x in fnames]
    data = load_czi(fpaths)
    print('Loaded from', SubDsetType.name(dset_subtype), datadir, data.shape)
    if dset_subtype == SubDsetType.Iba1Ki64:
        # We just need the combined channel. we don't need the nuclear channel.
        # in order for the whole setup to work well, I'm just copying the channel twice.
        # when creating the input, the average of these channels will still be exactly this channel, which is what we want.
        # we want this channel as input to the network.
        # Note that mean and the stdev used to normalize this data will be different, but we can try to do that initially.
        data = data[..., 1:]
        data = np.tile(data, (1, 1, 1, 2))
    return data


if __name__ == '__main__':
    from ml_collections.config_dict import ConfigDict
    data_config = ConfigDict()
    data_config.subdset_type = SubDsetType.OnlyIba1
    datadir = '/Users/ashesh.ashesh/Documents/Datasets/HT_Stefania/20230327_Ki67_and_Iba1_trainingdata/'
    data = get_train_val_data(datadir, data_config, DataSplitType.Val, val_fraction=0.1, test_fraction=0.1)
