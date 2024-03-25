import os
from ast import literal_eval as make_tuple
from collections.abc import Sequence
from random import shuffle
from typing import List

import numpy as np

from denoisplit.core.custom_enum import Enum
from denoisplit.core.data_split_type import DataSplitType, get_datasplit_tuples
from denoisplit.core.tiff_reader import load_tiff


class TwoChannelData(Sequence):
    """
    each element in data_arr should be a N*H*W array
    """

    def __init__(self, data_arr1, data_arr2, paths_data1=None, paths_data2=None):
        assert len(data_arr1) == len(data_arr2)
        self.paths1 = paths_data1
        self.paths2 = paths_data2

        self._data = []
        for i in range(len(data_arr1)):
            assert data_arr1[i].shape == data_arr2[i].shape
            assert len(
                data_arr1[i].shape) == 3, f'Each element in data arrays should be a N*H*W, but {data_arr1[i].shape}'
            self._data.append(np.concatenate([data_arr1[i][..., None], data_arr2[i][..., None]], axis=-1))

    def __len__(self):
        n = 0
        for x in self._data:
            n += x.shape[0]
        return n

    def __getitem__(self, idx):
        n = 0
        for dataidx, x in enumerate(self._data):
            if idx < n + x.shape[0]:
                if self.paths1 is None:
                    return x[idx - n], None
                else:
                    return x[idx - n], (self.paths1[dataidx], self.paths2[dataidx])
            n += x.shape[0]
        raise IndexError('Index out of range')


class MultiChannelData(Sequence):
    """
    each element in data_arr should be a N*H*W array
    """

    def __init__(self, data_arr, paths=None):
        self.paths = paths

        self._data = data_arr

    def __len__(self):
        n = 0
        for x in self._data:
            n += x.shape[0]
        return n

    def __getitem__(self, idx):
        n = 0
        for dataidx, x in enumerate(self._data):
            if idx < n + x.shape[0]:
                if self.paths is None:
                    return x[idx - n], None
                else:
                    return x[idx - n], (self.paths[dataidx])
            n += x.shape[0]
        raise IndexError('Index out of range')


class SubDsetType(Enum):
    TwoChannel = 0
    OneChannel = 1
    MultiChannel = 2


def subset_data(dataA, dataB, dataidx_list):
    dataidx_list = sorted(dataidx_list)
    subset_dataA = []
    subset_dataB = [] if dataB is not None else None
    cur_dataidx = 0
    cumulative_datacount = 0
    for arr_idx in range(len(dataA)):
        for data_idx in range(len(dataA[arr_idx])):
            cumulative_datacount += 1
            if dataidx_list[cur_dataidx] == cumulative_datacount - 1:
                subset_dataA.append(dataA[arr_idx][data_idx:data_idx + 1])
                if dataB is not None:
                    subset_dataB.append(dataB[arr_idx][data_idx:data_idx + 1])
                cur_dataidx += 1
            if cur_dataidx >= len(dataidx_list):
                break
        if cur_dataidx >= len(dataidx_list):
            break
    return subset_dataA, subset_dataB


def get_train_val_data(datadir,
                       data_config,
                       datasplit_type: DataSplitType,
                       get_multi_channel_files_fn,
                       load_data_fn=None,
                       val_fraction=None,
                       test_fraction=None):
    dset_subtype = data_config.subdset_type
    if load_data_fn is None:
        load_data_fn = load_tiff

    if dset_subtype == SubDsetType.TwoChannel:
        fnamesA, fnamesB = get_multi_channel_files_fn()
        fpathsA = [os.path.join(datadir, x) for x in fnamesA]
        fpathsB = [os.path.join(datadir, x) for x in fnamesB]
        dataA = [load_data_fn(fpath) for fpath in fpathsA]
        dataB = [load_data_fn(fpath) for fpath in fpathsB]
    elif dset_subtype == SubDsetType.OneChannel:
        fnamesmixed = get_multi_channel_files_fn()
        fpathsmixed = [os.path.join(datadir, x) for x in fnamesmixed]
        fpathsA = fpathsB = fpathsmixed
        dataA = [load_data_fn(fpath) for fpath in fpathsmixed]
        # Note that this is important. We need to ensure that the sum of the two channels is the same as sum of these two channels.
        dataA = [x / 2 for x in dataA]
        dataB = [x.copy() for x in dataA]
    elif dset_subtype == SubDsetType.MultiChannel:
        fnamesA = get_multi_channel_files_fn()
        fpathsA = [os.path.join(datadir, x) for x in fnamesA]
        dataA = [load_data_fn(fpath) for fpath in fpathsA]
        fnamesB = None
        fpathsB = None
        dataB = None

    if dataB is not None:
        assert len(dataA) == len(dataB)
        for i in range(len(dataA)):
            assert dataA[i].shape == dataB[
                i].shape, f'{dataA[i].shape} != {dataB[i].shape}, {fpathsA[i]} != {fpathsB[i]} in shape'

            if len(dataA[i].shape) == 2:
                dataA[i] = dataA[i][None]
                dataB[i] = dataB[i][None]

    count = np.sum([x.shape[0] for x in dataA])
    framewise_fpathsA = []
    for onedata_A, onepath_A in zip(dataA, fpathsA):
        framewise_fpathsA += [onepath_A] * onedata_A.shape[0]

    framewise_fpathsB = None
    if dataB is not None:
        framewise_fpathsB = []
        for onedata_B, onepath_B in zip(dataB, fpathsB):
            framewise_fpathsB += [onepath_B] * onedata_B.shape[0]

    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, count)

    if datasplit_type == DataSplitType.All:
        pass
    elif datasplit_type == DataSplitType.Train:
        # print(train_idx)
        dataA, dataB = subset_data(dataA, dataB, train_idx)
        framewise_fpathsA = [framewise_fpathsA[i] for i in train_idx]
        if dataB is not None:
            framewise_fpathsB = [framewise_fpathsB[i] for i in train_idx]
    elif datasplit_type == DataSplitType.Val:
        # print(val_idx)
        dataA, dataB = subset_data(dataA, dataB, val_idx)
        framewise_fpathsA = [framewise_fpathsA[i] for i in val_idx]
        if dataB is not None:
            framewise_fpathsB = [framewise_fpathsB[i] for i in val_idx]
    elif datasplit_type == DataSplitType.Test:
        # print(test_idx)
        dataA, dataB = subset_data(dataA, dataB, test_idx)
        framewise_fpathsA = [framewise_fpathsA[i] for i in test_idx]
        if dataB is not None:
            framewise_fpathsB = [framewise_fpathsB[i] for i in test_idx]
    else:
        raise Exception("invalid datasplit")

    if dset_subtype == SubDsetType.MultiChannel:
        data = MultiChannelData(dataA, paths=framewise_fpathsA)
    else:
        data = TwoChannelData(dataA, dataB, paths_data1=framewise_fpathsA, paths_data2=framewise_fpathsB)
    print('Loaded from', SubDsetType.name(dset_subtype), datadir, len(data))
    print('')
    return data
