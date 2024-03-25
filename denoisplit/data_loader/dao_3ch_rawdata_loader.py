import os
from ast import literal_eval as make_tuple
from collections.abc import Sequence
from random import shuffle
from typing import List

import numpy as np

from denoisplit.core.custom_enum import Enum
from denoisplit.core.data_split_type import DataSplitType, get_datasplit_tuples
from denoisplit.core.tiff_reader import load_tiff
from denoisplit.data_loader.multifile_raw_dloader import SubDsetType
from denoisplit.data_loader.multifile_raw_dloader import get_train_val_data as get_train_val_data_twochannels


def get_multi_channel_files():
    return ['reduced_SIM1-100.tif', 'reduced_SIM101-200.tif', 'reduced_SIM201-263.tif']


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    assert data_config.subdset_type == SubDsetType.MultiChannel
    return get_train_val_data_twochannels(datadir,
                                          data_config,
                                          datasplit_type,
                                          get_multi_channel_files,
                                          val_fraction=val_fraction,
                                          test_fraction=test_fraction)


if __name__ == '__main__':
    from denoisplit.data_loader.multifile_raw_dloader import SubDsetType
    from ml_collections.config_dict import ConfigDict
    data_config = ConfigDict()
    data_config.subdset_type = SubDsetType.MultiChannel
    datadir = '/group/jug/ashesh/data/Dao3ChannelReduced/'
    data = get_train_val_data(datadir, data_config, DataSplitType.Train, val_fraction=0.1, test_fraction=0.1)
    print(len(data))
    for i in range(len(data)):
        print(i, data[i][0].shape)
