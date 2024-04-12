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


def get_just_input_files():
    """
    In this case, we don't have access to the target. we just have access to the input.
    """
    fnames = [
            #'realdata/182_ER099_D86_S24mGFP_Tilescan_TRITC_T02-crop2_upscaled_crop.tif',
            #'realdata/ER041_Tilescan_merged_RobbiDone-crop2.tif_Frame1_upscaled-crop1_z680-710-1.tif',
            #'realdata/ER041_Tilescan_merged_RobbiDone-crop3.tif_Frame1_upscaled-crop1-1.tif',
            'lowres/230812R10S24mGFP_Rx_sectionA2_B2.1_GFP488_post4xM_stack2.tif',
    ]
    return fnames

def get_two_channel_files():
    fnames = [
            '230730ER111S24mGFP_sectionB3_GFP488_post4xM_stack1_230725.tif',
            # '230812R10S24mGFP_Rx_sectionA2_B2.1_GFP488_post4xM_stack2.tif',
            # '240323ER111S24mGFP_GFP488_sectionB4_post4xM_stack6.tif',
            # '230730ER111S24mGFP_sectionB3_GFP488_post4xM_stack2_230725.tif',
            # '230812R10S24mGFP_Rx_sectionA2_B2.1_GFP488_post4xM_stack3.tif',
            # '240323ER111S24mGFP_GFP488_sectionB5_post4xM_stack1.tif',
            # '230730ER111S24mGFP_sectionB6_GFP488_post4xM_stack1_230725.tif',
            # '240321ER142S24mGFP_sectionA3_GFP488_post4xM_stack1.tif',
            # '240323ER111S24mGFP_GFP488_sectionB5_post4xM_stack2.tif',
            # '230730ER111S24mGFP_sectionB6_GFP488_post4xM_stack3_230725.tif',
            # '240321ER142S24mGFP_sectionA3_GFP488_post4xM_stack2.tif',
            # '240323ER111S24mGFP_GFP488_sectionB5_post4xM_stack3.tif',
            # '230812R10S24mGFP_Rx_sectionA1_B2.1_GFP488_post4xM_stack1.tif',
            # '240321ER142S24mGFP_sectionA3_GFP488_post4xM_stack3.tif',
            # '240323ER111S24mGFP_GFP488_sectionB5_post4xM_stack4.tif',
            # '230812R10S24mGFP_Rx_sectionA1_B2.1_GFP488_post4xM_stack2.tif',
            # '240321ER142S24mGFP_sectionA3_GFP488_post4xM_stack4.tif',
            # '240323ER111S24mGFP_GFP488_sectionB5_post4xM_stack5.tif',
            # '230812R10S24mGFP_Rx_sectionA1_B2.1_GFP488_post4xM_stack3.tif',
            # '240321ER142S24mGFP_sectionA3_GFP488_post4xM_stack5.tif',
            # '240323ER111S24mGFP_GFP488_sectionB5_post4xM_stack6.tif',
            # '230812R10S24mGFP_Rx_sectionA1_B2.1_GFP488_post4xM_stack4.tif',
            # '240321ER142S24mGFP_sectionA3_GFP488_post4xM_stack6.tif',
            # '240323SJS60S24mGFP_GFP488_sectionA5_post4xM_stack1.tif',
            # '230812R10S24mGFP_Rx_sectionA1_B2.1_GFP488_post4xM_stack5.tif',
            # '240321ER142S24mGFP_sectionA3_GFP488_post4xM_stack7.tif',
            # '240323SJS60S24mGFP_GFP488_sectionA5_post4xM_stack2.tif',
            # '230812R10S24mGFP_Rx_sectionA1_B2.1_GFP488_post4xM_stack6.tif',
            # '240323ER111S24mGFP_GFP488_sectionB4_post4xM_stack1.tif',
            # '240323SJS60S24mGFP_GFP488_sectionA5_post4xM_stack3.tif',
            # '230812R10S24mGFP_Rx_sectionA1_B2.1_GFP488_post4xM_stack7.tif',
            # '240323ER111S24mGFP_GFP488_sectionB4_post4xM_stack2.tif',
            # '240323SJS60S24mGFP_GFP488_sectionA5_post4xM_stack4.tif',
            # '230812R10S24mGFP_Rx_sectionA1_B2.1_GFP488_post4xM_stack8.tif',
            #'240323ER111S24mGFP_GFP488_sectionB4_post4xM_stack3.tif',
            # '240323SJS60S24mGFP_GFP488_sectionA5_post4xM_stack5.tif',
            # '230812R10S24mGFP_Rx_sectionA1_B2.1_GFP488_post4xM_stack9.tif',
            # '240323ER111S24mGFP_GFP488_sectionB4_post4xM_stack4.tif',
            # '240323SJS60S24mGFP_GFP488_sectionA5_post4xM_stack6.tif',
            #'230812R10S24mGFP_Rx_sectionA2_B2.1_GFP488_post4xM_stack1.tif',
            # '240323ER111S24mGFP_GFP488_sectionB4_post4xM_stack5.tif',
            # '240323SJS60S24mGFP_GFP488_sectionA5_post4xM_stack7.tif',
    ]
    lowres = [os.path.join('lowres', f) for f in fnames]
    highres = [os.path.join('highres', f) for f in fnames]
    return lowres, highres




def get_train_val_filenames(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    if data_config.subdset_type == SubDsetType.TwoChannel:
        files_fn = get_two_channel_files
    elif data_config.subdset_type == SubDsetType.OneChannel:
        files_fn = get_just_input_files
    return files_fn

def get_train_val_data(datadir, data_config, datasplit_type: DataSplitType, val_fraction=None, test_fraction=None):
    files_fn = get_train_val_filenames(datadir, data_config, datasplit_type, val_fraction, test_fraction)
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
    datadir = '/group/jug/ashesh/data/Ekin/'
    data = get_train_val_data(datadir, data_config, DataSplitType.Train, val_fraction=0.1, test_fraction=0.1)
    print(len(data))
    # for i in range(len(data)):
    # print(i, data[i].shape)
