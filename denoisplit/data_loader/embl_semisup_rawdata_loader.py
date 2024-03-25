"""

"""
from typing import Union

import numpy as np
import os
from denoisplit.core import data_split_type
from denoisplit.core.tiff_reader import load_tiff
from denoisplit.core.data_type import DataType
from denoisplit.core.data_split_type import DataSplitType


def get_random_datasplit_tuples(val_fraction, test_fraction, N):
    if test_fraction is None:
        test_fraction = 0.0

    idx_arr = np.random.RandomState(seed=955).permutation(np.arange(N))
    trainN = int((1 - val_fraction - test_fraction) * N)
    valN = int(val_fraction * N)
    return idx_arr[:trainN].copy(), idx_arr[trainN:trainN + valN].copy(), idx_arr[trainN + valN:].copy()


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    fpath_mix = os.path.join(datadir, data_config.mix_fpath)
    fpath_ch1 = os.path.join(datadir, data_config.ch1_fpath)
    print(f'Loading Mix:{fpath_mix} & Ch1:{fpath_ch1} datasplit mode:{DataSplitType.name(datasplit_type)}')

    data_mix = load_tiff(fpath_mix).astype(np.float32)
    data_ch1 = load_tiff(fpath_ch1).astype(np.float32)

    if datasplit_type == DataSplitType.All:
        return {'mix': data_mix, 'C1': data_ch1}

    assert len(data_mix) == len(data_ch1)
    # Here, we have a very clear distribution shift as we increase the index. So, best option is to random splitting.
    train_idx, val_idx, test_idx = get_random_datasplit_tuples(val_fraction, test_fraction, len(data_mix))

    if datasplit_type == DataSplitType.Train:
        return {'mix': data_mix[train_idx], 'C1': data_ch1[train_idx]}
    elif datasplit_type == DataSplitType.Val:
        return {'mix': data_mix[val_idx], 'C1': data_ch1[val_idx]}
    elif datasplit_type == DataSplitType.Test:
        return {'mix': data_mix[test_idx], 'C1': data_ch1[test_idx]}
    else:
        raise Exception("invalid datasplit")
