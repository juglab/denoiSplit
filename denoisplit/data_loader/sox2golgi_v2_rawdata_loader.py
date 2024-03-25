import os
from functools import partial

import numpy as np

from denoisplit.core.data_split_type import DataSplitType
from denoisplit.data_loader.multifile_raw_dloader import get_train_val_data as get_train_val_data_multichannel
from nd2reader import ND2Reader


def get_start_end_index(key):
    """
    Few start and end frames are not good in some of the files. So, we need to exclude them.
    """
    start_index_dict = {
        'Test1_Slice1/1.nd2': 8,
        'Test1_Slice1/2.nd2': 1,
        'Test1_Slice1/3.nd2': 3,
        'Test1_Slice2_a/4.nd2': 10,
        'Test1_Slice2_a/5.nd2': 10,
        'Test1_Slice2_a/6.nd2': 10,
        'Test1_Slice2_b/7.nd2': 1,
        'Test1_Slice3_b/4.nd2': 1,
        'Test1_Slice3_b/5.nd2': 1,
        'Test1_Slice3_b/6.nd2': 1,
        'Test1_Slice4_a/1.nd2': 1,
        'Test1_Slice4_a/2.nd2': 1,
        'Test1_Slice4_a/3.nd2': 1,
        'Test1_Slice4_b/4.nd2': 1,
        'Test1_Slice4_b/5.nd2': 1,
        'Test1_Slice4_b/6.nd2': 1,
    }
    # excluding this index
    end_index_dict = {
        'Test1_Slice2_b/7.nd2': 18,
        'Test1_Slice2_b/8.nd2': 18,
        'Test1_Slice2_b/9.nd2': 18,
        'Test1_Slice3_a/1.nd2': 15,
        'Test1_Slice3_a/2.nd2': 15,
        'Test1_Slice3_a/3.nd2': 15,
        'Test1_Slice3_b/4.nd2': 18,
        'Test1_Slice3_b/5.nd2': 18,
        'Test1_Slice3_b/6.nd2': 18,
        'Test1_Slice4_a/1.nd2': 19,
        'Test1_Slice4_a/2.nd2': 19,
        'Test1_Slice4_a/3.nd2': 19,
    }
    return start_index_dict.get(key), end_index_dict.get(key)


def load_nd2(fpath, channel_names=None, multiplicative_factor=1):
    fname = os.path.basename(fpath)
    parent_dir = os.path.basename(os.path.dirname(fpath))
    key = os.path.join(parent_dir, fname)
    start_z, end_z = get_start_end_index(key)
    with ND2Reader(fpath) as reader:
        data = []
        if start_z is None:
            start_z = 0
        if end_z is None:
            end_z = reader.metadata['total_images_per_channel']

        all_channels = reader.metadata['channels']
        relevant_channel_indices = [all_channels.index(c) for c in channel_names]

        for z in range(start_z, end_z):
            channels = []
            for c in relevant_channel_indices:
                img = reader.get_frame_2D(c=c, z=z)
                img = img * multiplicative_factor
                channels.append(img[..., None])
            img = np.concatenate(channels, axis=-1)
            data.append(img[None])
        data = np.concatenate(data, axis=0)
    return data


def get_files():
    rel_fpaths = []
    rel_fpaths += ['Test1_Slice1/1.nd2', 'Test1_Slice1/2.nd2', 'Test1_Slice1/3.nd2']
    rel_fpaths += ['Test1_Slice2_a/4.nd2', 'Test1_Slice2_a/5.nd2', 'Test1_Slice2_a/6.nd2']
    rel_fpaths += ['Test1_Slice2_b/7.nd2', 'Test1_Slice2_b/8.nd2', 'Test1_Slice2_b/9.nd2']
    rel_fpaths += ['Test1_Slice3_a/1.nd2', 'Test1_Slice3_a/2.nd2', 'Test1_Slice3_a/3.nd2']
    rel_fpaths += ['Test1_Slice3_b/4.nd2', 'Test1_Slice3_b/5.nd2', 'Test1_Slice3_b/6.nd2']
    rel_fpaths += ['Test1_Slice4_a/1.nd2', 'Test1_Slice4_a/2.nd2', 'Test1_Slice4_a/3.nd2']
    rel_fpaths += ['Test1_Slice4_b/4.nd2', 'Test1_Slice4_b/5.nd2', 'Test1_Slice4_b/6.nd2']
    return rel_fpaths


def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    channel_names = [data_config.channel_1,
                     data_config.channel_2]  # There are 3 channels ['555-647', 'GT_Cy5', 'GT_TRITC']
    load_data_fn = partial(load_nd2, channel_names=channel_names)

    if set(channel_names) == set(['555-647']) and data_config.input_is_sum == False:
        # input is (C1 + C2 )/2. So, we need to divide by 2 for the input as well
        load_data_fn = partial(load_nd2, channel_names=channel_names, multiplicative_factor=0.5)

    print(
        f'Loading data from {datadir} with channel names {channel_names}, datasplit_type {DataSplitType.name(datasplit_type)}'
    )
    return get_train_val_data_multichannel(datadir,
                                           data_config,
                                           datasplit_type,
                                           get_files,
                                           load_data_fn=partial(load_nd2, channel_names=channel_names),
                                           val_fraction=val_fraction,
                                           test_fraction=test_fraction)


if __name__ == '__main__':
    import ml_collections
    from denoisplit.data_loader.multifile_raw_dloader import SubDsetType

    config = ml_collections.ConfigDict()
    config.subdset_type = SubDsetType.MultiChannel
    config.channel_1 = 'GT_Cy5'
    config.channel_2 = 'GT_TRITC'
    data = get_train_val_data('/group/jug/ashesh/data/TavernaSox2Golgi/acquisition2/',
                              config,
                              DataSplitType.Train,
                              val_fraction=0.1,
                              test_fraction=0.1)
    print(len(data))
