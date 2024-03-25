import os
from ast import literal_eval as make_tuple
from collections.abc import Sequence
from random import shuffle
from typing import List

import numpy as np

from czifile import imread as imread_czi
from denoisplit.core.custom_enum import Enum
from denoisplit.core.data_split_type import DataSplitType, get_datasplit_tuples
from denoisplit.core.tiff_reader import load_tiff
from denoisplit.data_loader.multifile_raw_dloader import SubDsetType
from denoisplit.data_loader.multifile_raw_dloader import get_train_val_data as get_train_val_data_twochannels


def get_multi_channel_files():
    return [
        'Experiment-447.czi',
        'Experiment-449.czi',
        'Experiment-448.czi',
        # 'Experiment-452.czi'
    ]


def load_data(fpath):
    # (4, 1, 4, 22, 512, 512, 1)
    data = imread_czi(fpath)
    clean_data = data[3, 0, [0, 2], ..., 0]
    clean_data = np.swapaxes(clean_data[..., None], 0, 4)[0]
    return clean_data


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    assert data_config.subdset_type == SubDsetType.MultiChannel
    return get_train_val_data_twochannels(datadir,
                                          data_config,
                                          datasplit_type,
                                          get_multi_channel_files,
                                          load_data_fn=load_data,
                                          val_fraction=val_fraction,
                                          test_fraction=test_fraction)


if __name__ == '__main__':
    from denoisplit.data_loader.multifile_raw_dloader import SubDsetType
    from ml_collections.config_dict import ConfigDict
    data_config = ConfigDict()
    data_config.subdset_type = SubDsetType.MultiChannel
    datadir = '/group/jug/ashesh/data/expansion_microscopy_v2/'
    data = get_train_val_data(datadir, data_config, DataSplitType.Train, val_fraction=0.1, test_fraction=0.1)
    print(len(data))
    for i in range(len(data)):
        print(i, data[i][0].shape)
