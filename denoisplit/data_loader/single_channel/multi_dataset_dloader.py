"""
If one has multiple .tif files, each corresponding to a different hardware setting. 
In this case, one needs to normalize these separate files separately.
"""
import ml_collections
import torch
import enum
from typing import Union, Tuple
import numpy as np

from denoisplit.data_loader.patch_index_manager import GridIndexManager, GridAlignement
from denoisplit.core import data_split_type
from denoisplit.core.data_split_type import DataSplitType
from denoisplit.data_loader.single_channel.single_channel_dloader import SingleChannelDloader
from denoisplit.data_loader.single_channel.single_channel_mc_dloader import SingleChannelMSDloader


class SingleChannelMultiDatasetDloader:

    def __init__(self,
                 data_config,
                 fpath: str,
                 datasplit_type: DataSplitType = None,
                 val_fraction=None,
                 test_fraction=None,
                 normalized_input=None,
                 enable_rotation_aug: bool = False,
                 enable_random_cropping: bool = False,
                 use_one_mu_std=None,
                 num_scales=None,
                 padding_kwargs: dict = None,
                 allow_generation=False,
                 max_val=None) -> None:

        assert isinstance(data_config.mix_fpath_list, tuple) or isinstance(data_config.mix_fpath_list, list)
        self._dsets = []
        self._channelwise_quantile = data_config.get('channelwise_quantile', False)

        for i, fpath_tuple in enumerate(zip(data_config.mix_fpath_list, data_config.ch1_fpath_list)):
            new_data_config = ml_collections.ConfigDict(data_config)
            new_data_config.mix_fpath = fpath_tuple[0]
            new_data_config.ch1_fpath = fpath_tuple[1]
            if num_scales is None:
                dset = SingleChannelDloader(new_data_config,
                                            fpath,
                                            datasplit_type=datasplit_type,
                                            val_fraction=val_fraction,
                                            test_fraction=test_fraction,
                                            normalized_input=normalized_input,
                                            enable_rotation_aug=enable_rotation_aug,
                                            enable_random_cropping=enable_random_cropping,
                                            use_one_mu_std=use_one_mu_std,
                                            allow_generation=allow_generation,
                                            max_val=max_val[i] if max_val is not None else None)
            else:
                dset = SingleChannelMSDloader(new_data_config,
                                              fpath,
                                              datasplit_type=datasplit_type,
                                              val_fraction=val_fraction,
                                              test_fraction=test_fraction,
                                              normalized_input=normalized_input,
                                              enable_rotation_aug=enable_rotation_aug,
                                              enable_random_cropping=enable_random_cropping,
                                              use_one_mu_std=use_one_mu_std,
                                              allow_generation=allow_generation,
                                              num_scales=num_scales,
                                              padding_kwargs=padding_kwargs,
                                              max_val=max_val[i] if max_val is not None else None)
            self._dsets.append(dset)
        self._img_sz = self._dsets[0]._img_sz
        self._grid_sz = self._dsets[0]._grid_sz

    def get_data_shape(self):
        N = 0
        default_shape = list(self._dsets[0]._data.shape)
        for dset in self._dsets:
            N += dset._data.shape[0]

        default_shape[0] = N
        return tuple(default_shape)

    def compute_mean_std(self, allow_for_validation_data=False):
        mean_arr = []
        std_arr = []
        for dset in self._dsets:
            mean, std = dset.compute_mean_std(allow_for_validation_data=allow_for_validation_data)
            mean_arr.append(mean[None])
            std_arr.append(std[None])

        mean_vec = np.concatenate(mean_arr, axis=0)
        std_vec = np.concatenate(std_arr, axis=0)
        return mean_vec, std_vec

    def compute_individual_mean_std(self):
        mean_arr = []
        std_arr = []
        for i, dset in enumerate(self._dsets):
            mean_, std_ = dset.compute_individual_mean_std()
            mean_arr.append(mean_[None])
            std_arr.append(std_[None])
        return np.concatenate(mean_arr, axis=0), np.concatenate(std_arr, axis=0)

    def get_mean_std(self):
        mean_arr = []
        std_arr = []
        for i, dset in enumerate(self._dsets):
            mean_, std_ = dset.get_mean_std()
            mean_arr.append(mean_[None])
            std_arr.append(std_[None])
        return np.concatenate(mean_arr, axis=0), np.concatenate(std_arr, axis=0)

    def set_mean_std(self, mean_val, std_val):
        for i, dset in enumerate(self._dsets):
            dset.set_mean_std(mean_val[i], std_val[i])

    def set_img_sz(self, image_size, grid_size, alignment=GridAlignement.LeftTop):
        self._img_sz = image_size
        self._grid_sz = grid_size
        self.idx_manager = GridIndexManager(self.get_data_shape(), self._grid_sz, self._img_sz, alignment)
        for dset in self._dsets:
            dset.set_img_sz(image_size, grid_size, alignment=alignment)

    def get_max_val(self):
        max_val_arr = []
        for dset in self._dsets:
            max_val = dset.get_max_val()
            if self._channelwise_quantile:
                max_val_arr.append(np.array(max_val)[None])
            else:
                max_val_arr.append(max_val)

        if self._channelwise_quantile:
            # 2D
            return np.concatenate(max_val_arr, axis=0)
        else:
            # 1D
            return np.array(max_val_arr)

    def set_max_val(self, max_val):
        for i, dset in enumerate(self._dsets):
            dset.set_max_val(max_val[i])

    def _get_dataset_index(self, index):
        cum_index = 0
        for i, dset in enumerate(self._dsets):
            if index < cum_index + len(dset):
                return i, index - cum_index
            cum_index += len(dset)
        raise ValueError('Too large index:', index)

    def __getitem__(self, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        dset_index, data_index = self._get_dataset_index(index)
        output = (*self._dsets[dset_index][data_index], dset_index)
        assert len(output) == 3
        return output

    def __len__(self):
        tot_len = 0
        for dset in self._dsets:
            tot_len += len(dset)
        return tot_len


if __name__ == '__main__':
    from denoisplit.configs.semi_supervised_config import get_config
    config = get_config()
    datadir = '/group/jug/ashesh/data/EMBL_halfsupervised/Demixing_3P/'
    val_fraction = 0.1
    test_fraction = 0.1

    dset = SingleChannelMultiDatasetDloader(config.data,
                                            datadir,
                                            datasplit_type=DataSplitType.Train,
                                            val_fraction=val_fraction,
                                            test_fraction=test_fraction,
                                            normalized_input=config.data.normalized_input,
                                            enable_rotation_aug=False,
                                            enable_random_cropping=False,
                                            use_one_mu_std=config.data.use_one_mu_std,
                                            allow_generation=False,
                                            max_val=None)

    mean_val, std_val = dset.compute_mean_std()
    dset.set_mean_std(mean_val, std_val)
    inp, tar, dset_index = dset[0]
    print(inp.shape, tar.shape, dset_index)
