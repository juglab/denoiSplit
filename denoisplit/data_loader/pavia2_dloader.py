import numpy as np
import torch

import ml_collections
from denoisplit.core.data_split_type import DataSplitType
from denoisplit.data_loader.lc_multich_dloader import LCMultiChDloader
from denoisplit.data_loader.patch_index_manager import GridIndexManager
from denoisplit.data_loader.pavia2_enums import Pavia2BleedthroughType
from denoisplit.data_loader.pavia2_rawdata_loader import Pavia2DataSetChannels, Pavia2DataSetType
from denoisplit.data_loader.vanilla_dloader import MultiChDloader


class Pavia2V1Dloader:

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
                 allow_generation=False,
                 max_val=None) -> None:

        self._datasplit_type = datasplit_type
        self._enable_random_cropping = enable_random_cropping
        self._dloader_clean = self._dloader_bleedthrough = self._dloader_mix = None
        self._use_one_mu_std = use_one_mu_std

        self._mean = None
        self._std = None
        assert normalized_input is True, "We are doing the normalization in this dataloader.So you better pass it as True"
        # We don't normalalize inside the self._dloader_clean or bleedthrough. We normalize in this class.
        normalized_input = False
        use_LC = 'multiscale_lowres_count' in data_config and data_config.multiscale_lowres_count is not None
        data_class = LCMultiChDloader if use_LC else MultiChDloader

        kwargs = {
            'normalized_input': normalized_input,
            'enable_rotation_aug': enable_rotation_aug,
            'use_one_mu_std': use_one_mu_std,
            'allow_generation': allow_generation,
            'datasplit_type': datasplit_type
        }
        if use_LC:
            padding_kwargs = {'mode': data_config.padding_mode}
            if 'padding_value' in data_config and data_config.padding_value is not None:
                padding_kwargs['constant_values'] = data_config.padding_value
            kwargs['padding_kwargs'] = padding_kwargs
            kwargs['num_scales'] = data_config.multiscale_lowres_count

        if self._datasplit_type == DataSplitType.Train:
            # assert enable_random_cropping is True
            dconf = ml_collections.ConfigDict(data_config)
            # take channels mean from this.
            dconf.dset_type = Pavia2DataSetType.JustMAGENTA
            self._clean_prob = dconf.dset_clean_sample_probab
            self._bleedthrough_prob = dconf.dset_bleedthrough_sample_probab
            assert self._clean_prob + self._bleedthrough_prob <= 1
            self._dloader_clean = data_class(dconf,
                                             fpath,
                                             val_fraction=val_fraction,
                                             test_fraction=test_fraction,
                                             enable_random_cropping=True,
                                             max_val=None,
                                             **kwargs)

            dconf.dset_type = Pavia2DataSetType.JustCYAN
            self._dloader_bleedthrough = data_class(dconf,
                                                    fpath,
                                                    val_fraction=val_fraction,
                                                    test_fraction=test_fraction,
                                                    enable_random_cropping=True,
                                                    max_val=None,
                                                    **kwargs)

            dconf.dset_type = Pavia2DataSetType.MIXED
            self._dloader_mix = data_class(dconf,
                                           fpath,
                                           val_fraction=val_fraction,
                                           test_fraction=test_fraction,
                                           enable_random_cropping=True,
                                           max_val=None,
                                           **kwargs)
        else:
            assert enable_random_cropping is False
            dconf = ml_collections.ConfigDict(data_config)
            dconf.dset_type = Pavia2DataSetType.JustMAGENTA
            # we want to evaluate on mixed samples.
            self._clean_prob = 1.0
            self._bleedthrough_prob = 0.0
            self._dloader_clean = data_class(dconf,
                                             fpath,
                                             val_fraction=val_fraction,
                                             test_fraction=test_fraction,
                                             enable_random_cropping=enable_random_cropping,
                                             max_val=max_val,
                                             **kwargs)
        self.process_data()

        # needed just during evaluation.
        self._img_sz = self._dloader_clean._img_sz
        self._grid_sz = self._dloader_clean._grid_sz

        print(f'[{self.__class__.__name__}] BleedTh prob:{self._bleedthrough_prob} Clean prob:{self._clean_prob}')

    def sum_channels(self, data, first_index_arr, second_index_arr):
        fst_channel = data[..., first_index_arr].sum(axis=-1, keepdims=True)
        scnd_channel = data[..., second_index_arr].sum(axis=-1, keepdims=True)
        return np.concatenate([fst_channel, scnd_channel], axis=-1)

    def process_data(self):
        """
        We are ignoring the actin channel.
        We know that MTORQ(uise) has sigficant bleedthrough from TUBULIN channels. So, when MTORQ has no content, then 
        we sum it with TUBULIN so that tubulin has whole of its content. 
        When MTORQ has content, then we sum RFP670 with tubulin. This makes sure that tubulin channel has the same data distribution. 
        During validation/testing, we always feed sum of these three channels as the input.
        """

        if self._datasplit_type == DataSplitType.Train:
            self._dloader_clean._data = self._dloader_clean._data[
                ..., [Pavia2DataSetChannels.NucRFP670, Pavia2DataSetChannels.TUBULIN]]
            self._dloader_bleedthrough._data = self._dloader_bleedthrough._data[
                ..., [Pavia2DataSetChannels.NucMTORQ, Pavia2DataSetChannels.TUBULIN]]
            self._dloader_mix._data = self._dloader_mix._data[
                ..., [Pavia2DataSetChannels.NucRFP670, Pavia2DataSetChannels.NucMTORQ, Pavia2DataSetChannels.TUBULIN]]
            self._dloader_mix._data = self.sum_channels(self._dloader_mix._data, [0, 1], [2])
            self._dloader_mix._data[..., 0] = self._dloader_mix._data[..., 0] / 2
            # self._dloader_clean._data = self.sum_channels(self._dloader_clean._data, [1], [0, 2])
            # In bleedthrough dataset, the nucleus channel is empty.
            # self._dloader_bleedthrough._data = self.sum_channels(self._dloader_bleedthrough._data, [0], [1, 2])
        else:
            self._dloader_mix._data = self._dloader_mix._data[
                ..., [Pavia2DataSetChannels.NucRFP670, Pavia2DataSetChannels.NucMTORQ, Pavia2DataSetChannels.TUBULIN]]
            self._dloader_mix._data = self.sum_channels(self._dloader_mix._data, [0, 1], [2])

    def set_img_sz(self, image_size, grid_size, alignment=None):
        """
        Needed just for the notebooks
        If one wants to change the image size on the go, then this can be used.
        Args:
            image_size: size of one patch
            grid_size: frame is divided into square grids of this size. A patch centered on a grid having size `image_size` is returned.
        """
        self._img_sz = image_size
        self._grid_sz = grid_size
        if self._dloader_mix is not None:
            self._dloader_mix.set_img_sz(image_size, grid_size, alignment=alignment)

        if self._dloader_clean is not None:
            self._dloader_clean.set_img_sz(image_size, grid_size, alignment=alignment)

        if self._dloader_bleedthrough is not None:
            self._dloader_bleedthrough.set_img_sz(image_size, grid_size, alignment=alignment)

        self.idx_manager = GridIndexManager(self.get_data_shape(), self._grid_sz, self._img_sz, alignment)

    def get_mean_std(self):
        """
        Needed just for running the notebooks
        """
        return self._mean, self._std

    def get_data_shape(self):
        N = 0
        default_shape = None
        if self._dloader_mix is not None:
            default_shape = self._dloader_mix.get_data_shape()
            N += default_shape[0]

        if self._dloader_clean is not None:
            default_shape = self._dloader_clean.get_data_shape()
            N += default_shape[0]

        if self._dloader_bleedthrough is not None:
            default_shape = self._dloader_bleedthrough.get_data_shape()
            N += default_shape[0]

        default_shape = list(default_shape)
        default_shape[0] = N
        return tuple(default_shape)

    def __len__(self):
        sz = 0
        if self._dloader_clean is not None:
            sz += int(self._clean_prob * len(self._dloader_clean))
        if self._dloader_bleedthrough is not None:
            sz += int(self._bleedthrough_prob * len(self._dloader_bleedthrough))
        if self._dloader_mix is not None:
            mix_prob = 1 - self._clean_prob - self._bleedthrough_prob
            sz += int(mix_prob * len(self._dloader_mix))
        return sz

    def compute_individual_mean_std(self):
        mean_, std_ = self._dloader_clean.compute_individual_mean_std()
        mean_dict = {'target': mean_, 'mix': mean_.sum(axis=1, keepdims=True)}
        std_dict = {'target': std_, 'mix': np.sqrt((std_**2).sum(axis=1, keepdims=True))}
        # NOTE: dataloader2 does not has clean channel. So, no mean should be computed on it.
        # mean_std2 = self._dloader_bleedthrough.compute_individual_mean_std() if self._dloader_bleedthrough is not None else (None,None)
        return mean_dict, std_dict

        # if mean_std2 is None:
        #     return mean_std1

        # mean_val = (mean_std1[0] + mean_std2[0]) / 2
        # std_val = (mean_std1[1] + mean_std2[1]) / 2

        # return (mean_val, std_val)

    def compute_mean_std(self):
        if self._use_one_mu_std is False:
            return self.compute_individual_mean_std()
        else:
            raise ValueError('This must not be called. We want to compute individual mean so that they can be \
                passed on to the model')
            mean_std1 = self._dloader_clean.compute_mean_std()
            mean_std2 = self._dloader2.compute_mean_std() if self._dloader_bleedthrough is not None else (None, None)
            if mean_std2 is None:
                return mean_std1

            mean_val = (mean_std1[0] + mean_std2[0]) / 2
            std_val = (mean_std1[1] + mean_std2[1]) / 2

            return (mean_val, std_val)

    def set_mean_std(self, mean_val, std_val):
        self._mean = mean_val
        self._std = std_val

        # self._dloader_clean.set_mean_std(mean_val, std_val)
        # if self._dloader_bleedthrough is not None:
        #     self._dloader_bleedthrough.set_mean_std(mean_val, std_val)

    def normalize_input(self, inp):
        return (inp - self._mean['mix'][0]) / self._std['mix'][0]

    def __getitem__(self, index):
        """
        Returns:
            (inp,tar,mixed_recons_flag): When mixed_recons_flag is set, then do only the mixed reconstruction. This is set when we've bleedthrough
        """
        coin_flip = np.random.rand()
        if self._datasplit_type == DataSplitType.Train:

            if coin_flip <= self._clean_prob:
                idx = np.random.randint(len(self._dloader_clean))
                inp, tar = self._dloader_clean[idx]
                mixed_recons_flag = Pavia2BleedthroughType.Clean
                # print('Clean', idx)
            elif coin_flip > self._clean_prob and coin_flip <= self._clean_prob + self._bleedthrough_prob:
                idx = np.random.randint(len(self._dloader_bleedthrough))
                inp, tar = self._dloader_bleedthrough[idx]
                mixed_recons_flag = Pavia2BleedthroughType.Bleedthrough
                # print('Bleedthrough')
            else:
                idx = np.random.randint(len(self._dloader_mix))
                inp, tar = self._dloader_mix[idx]
                mixed_recons_flag = Pavia2BleedthroughType.Mixed
                # print('Mixed', idx)

            # dataloader takes the average of the K channels. To, undo that, we are multipying it with K.
            inp = len(tar) * inp
            inp = self.normalize_input(inp)
            return (inp, tar, mixed_recons_flag)

        else:
            inp, tar = self._dloader_clean[index]
            inp = len(tar) * inp
            inp = self.normalize_input(inp)
            return (inp, tar, Pavia2BleedthroughType.Clean)

    def get_max_val(self):
        max_val = self._dloader_clean.get_max_val()
        return max_val


if __name__ == '__main__':
    from denoisplit.configs.pavia2_config import get_config
    config = get_config()
    fpath = '/group/jug/ashesh/data/pavia2/'
    dloader = Pavia2V1Dloader(
        config.data,
        fpath,
        datasplit_type=DataSplitType.Val,
        val_fraction=0.1,
        test_fraction=0.1,
        normalized_input=True,
        use_one_mu_std=False,
        enable_random_cropping=False,
        max_val=100,
    )
    mean_val, std_val = dloader.compute_mean_std()
    dloader.set_mean_std(mean_val, std_val)
    inp, tar, source = dloader[0]
    len(dloader)
    print('This is working')