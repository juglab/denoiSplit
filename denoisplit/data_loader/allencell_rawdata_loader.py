import os
import numpy as np

from denoisplit.core.tiff_reader import load_tiffs
from denoisplit.core.data_split_type import DataSplitType, get_datasplit_tuples


def get_train_val_datafiles(dirname, datasplit_type, val_fraction, test_fraction):
    fnames = [
        'AICS-11_0.ome.tif', 'AICS-11_1.ome.tif', 'AICS-11_2.ome.tif', 'AICS-11_3.ome.tif', 'AICS-11_4.ome.tif',
        'AICS-11_5.ome.tif', 'AICS-11_6.ome.tif', 'AICS-11_7.ome.tif', 'AICS-11_8.ome.tif', 'AICS-11_9.ome.tif',
        'AICS-11_10.ome.tif', 'AICS-11_11.ome.tif', 'AICS-11_12.ome.tif', 'AICS-11_13.ome.tif', 'AICS-11_14.ome.tif',
        'AICS-11_15.ome.tif', 'AICS-11_16.ome.tif', 'AICS-11_17.ome.tif', 'AICS-11_18.ome.tif', 'AICS-11_19.ome.tif',
        'AICS-11_20.ome.tif', 'AICS-11_21.ome.tif', 'AICS-11_22.ome.tif', 'AICS-11_23.ome.tif', 'AICS-11_24.ome.tif',
        'AICS-11_25.ome.tif', 'AICS-11_26.ome.tif', 'AICS-11_27.ome.tif', 'AICS-11_28.ome.tif', 'AICS-11_29.ome.tif',
        'AICS-11_30.ome.tif', 'AICS-11_31.ome.tif', 'AICS-11_32.ome.tif', 'AICS-11_33.ome.tif', 'AICS-11_34.ome.tif',
        'AICS-11_35.ome.tif', 'AICS-11_36.ome.tif', 'AICS-11_37.ome.tif', 'AICS-11_38.ome.tif', 'AICS-11_39.ome.tif',
        'AICS-11_40.ome.tif', 'AICS-11_41.ome.tif', 'AICS-11_42.ome.tif', 'AICS-11_43.ome.tif', 'AICS-11_44.ome.tif',
        'AICS-11_45.ome.tif', 'AICS-11_46.ome.tif', 'AICS-11_47.ome.tif', 'AICS-11_48.ome.tif', 'AICS-11_49.ome.tif',
        'AICS-11_50.ome.tif', 'AICS-11_51.ome.tif', 'AICS-11_52.ome.tif', 'AICS-11_53.ome.tif', 'AICS-11_54.ome.tif',
        'AICS-11_55.ome.tif', 'AICS-11_56.ome.tif', 'AICS-11_57.ome.tif', 'AICS-11_58.ome.tif'
    ]

    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(fnames))
    test_names = [fnames[x] for x in test_idx]
    train_names = [fnames[x] for x in train_idx]
    val_names = [fnames[x] for x in val_idx]

    if datasplit_type == DataSplitType.Train:
        return [os.path.join(dirname, fname) for fname in train_names]
    elif datasplit_type == DataSplitType.Val:
        return [os.path.join(dirname, fname) for fname in val_names]
    elif datasplit_type == DataSplitType.Test:
        return [os.path.join(dirname, fname) for fname in test_names]


def get_std_mask(data, quantile):
    std_arr = np.array([data[i].std() for i in range(len(data))])
    std_thresh = np.quantile(std_arr, quantile)
    ch_mask = std_arr >= std_thresh
    return ch_mask


def get_train_val_data(dirname, data_config, datasplit_type, val_fraction, test_fraction):
    fpaths = get_train_val_datafiles(dirname, datasplit_type, val_fraction, test_fraction)
    print(
        f'Loading {dirname} with Channels {data_config.channel_1},{data_config.channel_2}, Mode:{DataSplitType.name(datasplit_type)}'
    )
    data = load_tiffs(fpaths)[..., [data_config.channel_1, data_config.channel_2]]
    if 'ch1_frame_std_quantile' in data_config:
        q_ch1 = data_config.ch1_frame_std_quantile
        ch1_mask = get_std_mask(data[..., 0], q_ch1)

        q_ch2 = data_config.ch2_frame_std_quantile
        ch2_mask = get_std_mask(data[..., 1], q_ch2)
        mask = np.logical_or(ch1_mask, ch2_mask)
        print(f'Skipped {(~mask).sum()} entries. Picking {mask.sum()} entries')
        return data[mask].copy()
    return data