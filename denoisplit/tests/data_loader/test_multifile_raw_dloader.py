from unittest import mock

import numpy as np

import ml_collections
from denoisplit.core.data_split_type import DataSplitType
from denoisplit.data_loader.multifile_raw_dloader import SubDsetType
from denoisplit.data_loader.multifile_raw_dloader import get_train_val_data as get_train_val_data_twofiles


def get_two_channel_files():
    fnamesA = []
    fnamesB = []
    j = 1
    for val in range(100):
        sz = 512 if val % 3 == 0 else 256
        fnamesA.append(f'A_{val}_{j}_{sz}')
        fnamesB.append(f'B_{val}_{j}_{sz}')
        j += 1
        if j == 11:
            j = 1

    return fnamesA, fnamesB


def load_tiff_same_count(fpath):
    a_or_b, val, count, sz = fpath.split('_')
    val = int(val)
    count = 1
    sz = int(sz)
    val = val if a_or_b == 'A' else val * -1
    return np.ones((count, sz, sz)) * val


@mock.patch('disentangle.data_loader.multifile_raw_dloader.load_tiff', side_effect=load_tiff_same_count)
def test_multifile_raw_dloader(mock_load_tiff):
    config = ml_collections.ConfigDict()
    config.subdset_type = SubDsetType.TwoChannel
    data_test = get_train_val_data_twofiles('',
                                            config,
                                            DataSplitType.Test,
                                            get_two_channel_files,
                                            val_fraction=0.15,
                                            test_fraction=0.1)
    data_train = get_train_val_data_twofiles('',
                                             config,
                                             DataSplitType.Train,
                                             get_two_channel_files,
                                             val_fraction=0.15,
                                             test_fraction=0.1)
    data_val = get_train_val_data_twofiles('',
                                           config,
                                           DataSplitType.Val,
                                           get_two_channel_files,
                                           val_fraction=0.15,
                                           test_fraction=0.1)
    assert len(data_test) == 10
    assert len(data_train) == 75
    assert len(data_val) == 15

    train_unique = [np.unique(data_train[i][..., 0]).tolist() for i in range(len(data_train))]
    train_vals = []
    for elem in train_unique:
        assert len(elem) == 1
        train_vals.append(elem[0])
    assert len(train_vals) == len(set(train_vals))

    val_unique = [np.unique(data_val[i][..., 0]).tolist() for i in range(len(data_val))]
    val_vals = []
    for elem in val_unique:
        assert len(elem) == 1
        val_vals.append(elem[0])
    assert len(val_vals) == len(set(val_vals))

    test_unique = [np.unique(data_test[i][..., 0]).tolist() for i in range(len(data_test))]
    test_vals = []
    for elem in test_unique:
        assert len(elem) == 1
        test_vals.append(elem[0])
    assert len(test_vals) == len(set(test_vals))

    assert len(set(train_vals).intersection(set(val_vals))) == 0
    assert len(set(train_vals).intersection(set(test_vals))) == 0
    assert len(set(val_vals).intersection(set(test_vals))) == 0


def load_tiff_different_count(fpath):
    a_or_b, val, count, sz = fpath.split('_')
    val = int(val)
    count = int(count)
    sz = int(sz)
    val = val if a_or_b == 'A' else val * -1
    return np.ones((count, sz, sz)) * val


@mock.patch('disentangle.data_loader.multifile_raw_dloader.load_tiff', side_effect=load_tiff_different_count)
def test_multifile_raw_dloader(mock_load_tiff):
    config = ml_collections.ConfigDict()
    config.subdset_type = SubDsetType.TwoChannel
    data_test = get_train_val_data_twofiles('',
                                            config,
                                            DataSplitType.Test,
                                            get_two_channel_files,
                                            val_fraction=0.15,
                                            test_fraction=0.1)
    data_train = get_train_val_data_twofiles('',
                                             config,
                                             DataSplitType.Train,
                                             get_two_channel_files,
                                             val_fraction=0.15,
                                             test_fraction=0.1)
    data_val = get_train_val_data_twofiles('',
                                           config,
                                           DataSplitType.Val,
                                           get_two_channel_files,
                                           val_fraction=0.15,
                                           test_fraction=0.1)

    cnt = 0
    for fpath in get_two_channel_files()[0]:
        cnt += load_tiff_different_count(fpath).shape[0]

    assert abs(len(data_test) - int(cnt * 0.1)) < 2
    assert abs(len(data_train) - int(cnt * 0.75)) < 2
    assert abs(len(data_val) - int(cnt * 0.15)) < 2

    # make sure that the values of the two channels are in sync
    for i in range(len(data_train)):
        assert np.all(data_train[i][..., 0] == -1 * data_train[i][..., 1])

    train_unique = [np.unique(data_train[i][..., 0]).tolist() for i in range(len(data_train))]
    train_vals = []
    for elem in train_unique:
        assert len(elem) == 1
        train_vals.append(elem[0])

    val_unique = [np.unique(data_val[i][..., 0]).tolist() for i in range(len(data_val))]
    val_vals = []
    for elem in val_unique:
        assert len(elem) == 1
        val_vals.append(elem[0])

    test_unique = [np.unique(data_test[i][..., 0]).tolist() for i in range(len(data_test))]
    test_vals = []
    for elem in test_unique:
        assert len(elem) == 1
        test_vals.append(elem[0])

    all_vals = np.array(train_vals + val_vals + test_vals)

    for fpath in get_two_channel_files()[0]:
        val = int(fpath.split('_')[1])
        count = int(fpath.split('_')[2])
        assert np.sum(all_vals == val) == count
