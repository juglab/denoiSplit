import os

import numpy as np

from denoisplit.core.data_split_type import DataSplitType, get_datasplit_tuples
from denoisplit.core.tiff_reader import load_tiff


def get_data_from_paths(fpaths1, fpaths2, enable_max_projection=False):
    data1 = [load_tiff(path)[..., None] for path in fpaths1]

    data2 = [load_tiff(path)[..., None] for path in fpaths2]
    if enable_max_projection:
        data1 = [np.max(x, axis=1, keepdims=True) for x in data1]
        data2 = [np.max(x, axis=1, keepdims=True) for x in data2]

    # squishing the 1st and 2nd dimension.
    data1 = [x.reshape(np.prod(x.shape[:2]), *x.shape[2:]) for x in data1]
    data2 = [x.reshape(np.prod(x.shape[:2]), *x.shape[2:]) for x in data2]

    data1 = np.concatenate(data1, axis=0)
    data2 = np.concatenate(data2, axis=0)
    assert data1.shape[0] == data2.shape[0], 'For now, we need both channels to have identical data'
    data = np.concatenate([data1, data2], axis=3)
    return data


def get_train_val_data(dirname, data_config, datasplit_type, val_fraction, test_fraction):
    # actin-60x-noise2-highsnr.tif  mito-60x-noise2-highsnr.tif
    all_fpaths1 = [os.path.join(dirname, x) for x in mito_channel_fnames()]
    all_fpaths2 = [os.path.join(dirname, x) for x in er_channel_fnames()]

    assert len(all_fpaths1) == len(all_fpaths2), 'Currently, only same sized data in both channels is supported'

    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction,
                                                        test_fraction,
                                                        len(all_fpaths1),
                                                        starting_test=True)
    if datasplit_type == DataSplitType.Train:
        fpaths1 = [all_fpaths1[idx] for idx in train_idx]
        fpaths2 = [all_fpaths2[idx] for idx in train_idx]
    elif datasplit_type == DataSplitType.Val:
        fpaths1 = [all_fpaths1[idx] for idx in val_idx]
        fpaths2 = [all_fpaths2[idx] for idx in val_idx]
    elif datasplit_type == DataSplitType.Test:
        fpaths1 = [all_fpaths1[idx] for idx in test_idx]
        fpaths2 = [all_fpaths2[idx] for idx in test_idx]
    elif datasplit_type == DataSplitType.All:
        fpaths1 = all_fpaths1
        fpaths2 = all_fpaths2

    print(f'Loading from {dirname}, Mode:{DataSplitType.name(datasplit_type)}, PerChannelFilecount:{len(fpaths1)}')
    data = get_data_from_paths(fpaths1, fpaths2, enable_max_projection=data_config.enable_max_projection)
    return data


def mito_channel_fnames():
    # return [f'Mitotracker_Green_0{i}.tif' for i in [1,2,3,4,5,6]]
    return [f'Mitotracker_Green_0{i}.tif' for i in [1, 3, 4, 5, 6]]


def er_channel_fnames():
    return [f'ER-eGFP_only_0{i}.tif' for i in [1, 3, 4, 5, 6]]
