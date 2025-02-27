from denoisplit.core.tiff_reader import save_tiff
from denoisplit.core.data_type import DataType
from denoisplit.core.data_split_type import DataSplitType
from denoisplit.data_loader.two_tiff_rawdata_loader import get_train_val_data


import ml_collections 
config = ml_collections.ConfigDict()
config.data = ml_collections.ConfigDict()

config.data.data_type = DataType.SeparateTiffData
config.data.channel_1 = 0
config.data.channel_2 = 1
config.data.ch1_fname = 'actin-60x-noise2-lowsnr.tif'
config.data.ch2_fname = 'mito-60x-noise2-lowsnr.tif'

config.training = ml_collections.ConfigDict()
config.training.val_fraction = 0.1
config.training.test_fraction = 0.1


data = get_train_val_data('/group/jug/ashesh/data/ventura_gigascience/', config.data, DataSplitType.Test,
                            config.training.val_fraction, config.training.test_fraction)
save_tiff('/group/jug/ashesh/data/diffsplit_hagen/test/test_actin-60x-noise2-highsnr.tif', data[...,0])
save_tiff('/group/jug/ashesh/data/diffsplit_hagen/test/test_mito-60x-noise2-highsnr.tif', data[...,1])

data = get_train_val_data('/group/jug/ashesh/data/ventura_gigascience/', config.data, DataSplitType.Val,
                            config.training.val_fraction, config.training.test_fraction)
save_tiff('/group/jug/ashesh/data/diffsplit_hagen/val/val_actin-60x-noise2-highsnr.tif', data[...,0])
save_tiff('/group/jug/ashesh/data/diffsplit_hagen/val/val_mito-60x-noise2-highsnr.tif', data[...,1])

data = get_train_val_data('/group/jug/ashesh/data/ventura_gigascience/', config.data, DataSplitType.Train,
                            config.training.val_fraction, config.training.test_fraction)
save_tiff('/group/jug/ashesh/data/diffsplit_hagen/train/train_actin-60x-noise2-highsnr.tif', data[...,0])
save_tiff('/group/jug/ashesh/data/diffsplit_hagen/train/train_mito-60x-noise2-highsnr.tif', data[...,1])
