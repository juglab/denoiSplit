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


def get_two_channel_files():
    arr = [71, 89, 92, 93, 94, 95, 96, 97, 98, 99, 100, 1752, 1757, 1758, 1760, 1761]
    sox2 = [f'SOX2/C2-Experiment-{i}.tif' for i in arr]
    golgi = [f'GOLGI/C1-Experiment-{i}.tif' for i in arr]
    return sox2, golgi


def get_one_channel_files():
    c2exp = [1267, 1268, 1269, 1270, 1272, 1273, 1274]
    fpaths = [f'SOX2-Golgi/C2-Experiment-{i}.tif' for i in c2exp]

    c2osvz = [1294, 1295, 1296, 1297]
    fpaths += [f'SOX2-Golgi/C2-oSVZ-Experiment-{i}.tif' for i in c2osvz]

    c2Osvz = [1286, 1287]
    fpaths += [f'SOX2-Golgi/C2-OSVZ-Experiment-{i}.tif' for i in c2Osvz]

    c2svz = [1290, 1291, 1292, 1293]
    fpaths += [f'SOX2-Golgi/C2-SVZ-Experiment-{i}.tif' for i in c2svz]

    fpaths += [
        'SOX2-Golgi/C2-SVZ-Experiment-1282-Substack-9-12.tif', 'SOX2-Golgi/C2-SVZ-Experiment-1283-Substack-8-20.tif',
        'SOX2-Golgi/C2-SVZ-Experiment-1285-Substack-13-32.tif'
    ]
    return fpaths


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    if data_config.subdset_type == SubDsetType.OneChannel:
        files_fn = get_one_channel_files
    elif data_config.subdset_type == SubDsetType.TwoChannel:
        files_fn = get_two_channel_files

    return get_train_val_data_twochannels(datadir,
                                          data_config,
                                          datasplit_type,
                                          files_fn,
                                          val_fraction=val_fraction,
                                          test_fraction=test_fraction)


if __name__ == '__main__':
    from denoisplit.data_loader.multifile_raw_dloader import SubDsetType
    from ml_collections.config_dict import ConfigDict
    data_config = ConfigDict()
    data_config.subdset_type = SubDsetType.OneChannel
    datadir = '/group/jug/ashesh/data/TavernaSox2Golgi/'
    data = get_train_val_data(datadir, data_config, DataSplitType.Train, val_fraction=0.1, test_fraction=0.1)
    print(len(data))
    # for i in range(len(data)):
    # print(i, data[i].shape)
