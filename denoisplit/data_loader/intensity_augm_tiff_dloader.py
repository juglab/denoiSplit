"""
Here, the motivation is to have Intensity based augmentation We'll change the amount of the overlap in order for it.
"""
from typing import List, Tuple, Union

import numpy as np

from denoisplit.core.data_split_type import DataSplitType
from denoisplit.data_loader.vanilla_dloader import MultiChDloader


class Interval:

    def __init__(self, minv, maxv):
        self._minv = minv
        self._maxv = maxv

    def __contains__(self, val):
        lhs = self._minv < val
        rhs = val < self._maxv
        return lhs and rhs

    def sample(self):
        diff = self._maxv - self._minv
        return self._minv + np.random.rand() * diff


class AlphaClasses:
    """
    A class to sample alpha values. They will be used to compute the weighted average of the two channels.
    """

    def __init__(self, minv, maxv, nintervals=10):
        self._minv = minv
        self._maxv = maxv
        step = (self._maxv - self._minv) / nintervals
        self._intervals = []
        for minv_class in np.arange(self._minv, self._maxv + 1e-5, step):
            self._intervals.append(Interval(minv_class, minv_class + step))
        print(f'[{self.__class__.__name__}] {self._minv}-{self._maxv} {nintervals}')

    def class_ids(self):
        return list(range(len(self._intervals)))

    def sample(self, class_idx=None):
        if class_idx is not None:
            return self._intervals[class_idx].sample(), class_idx
        else:
            class_idx = np.random.randint(0, high=len(self._intervals))
            return self._intervals[class_idx].sample(), class_idx


class IntensityAugTiffDloader(MultiChDloader):

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
                 max_val=None):
        super().__init__(data_config,
                         fpath,
                         datasplit_type=datasplit_type,
                         val_fraction=val_fraction,
                         test_fraction=test_fraction,
                         normalized_input=normalized_input,
                         enable_rotation_aug=enable_rotation_aug,
                         enable_random_cropping=enable_random_cropping,
                         use_one_mu_std=use_one_mu_std,
                         allow_generation=allow_generation,
                         max_val=max_val)
        assert self._data.shape[-1] == 2
        self._ch1_min_alpha = data_config.ch1_min_alpha
        self._ch1_max_alpha = data_config.ch1_max_alpha
        self._ch1_alpha_interval_count = data_config.get('ch1_alpha_interval_count', 1)
        self._alpha_sampler = None
        self._cl_std_filter = data_config.get('cl_std_filter', None)

        if self._ch1_max_alpha is not None and self._ch1_min_alpha is not None:
            self._alpha_sampler = AlphaClasses(self._ch1_min_alpha,
                                               self._ch1_max_alpha,
                                               nintervals=self._ch1_alpha_interval_count)

        print(f'[{self.__class__.__name__}] CL_std_lowerb', self._cl_std_filter)
        # assert self._use_one_mu_std is False, "We need individual channels mean and std to be able to get correct mean for alpha alphas."

    def _sample_alpha(self, alpha_class_idx=None):
        if self._ch1_min_alpha is None or self._ch1_max_alpha is None:
            return None
        alpha, alpha_class_idx = self._alpha_sampler.sample(class_idx=alpha_class_idx)
        return alpha, alpha_class_idx

    def _compute_mean_std_with_alpha(self, alpha):
        mean, std = self.get_mean_std()
        mean = mean.squeeze()
        std = std.squeeze()
        mean = mean[0] * alpha + mean[1] * (1 - alpha)
        std = std[0] * alpha + std[1] * (1 - alpha)
        return mean, std

    def _compute_input_with_alpha(self, img_tuples, alpha, use_alpha_invariant_mean=False):
        assert len(img_tuples) == 2
        assert self._normalized_input is True, "normalization should happen here"

        inp = img_tuples[0] * alpha + img_tuples[1] * (1 - alpha)
        if use_alpha_invariant_mean:
            mean, std = self._compute_mean_std_with_alpha(0.5)
        else:
            mean, std = self._compute_mean_std_with_alpha(alpha)

        inp = (inp - mean) / std
        return inp.astype(np.float32)

    def _compute_input(self, img_tuples):
        alpha, _ = self._sample_alpha()
        assert alpha is not None
        return self._compute_input_with_alpha(img_tuples, alpha)


class IntensityAugCLTiffDloader(IntensityAugTiffDloader):
    """
    Dataset used in contrastive learning.
    """

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
                 return_individual_channels: bool = False,
                 return_alpha: bool = False,
                 use_alpha_invariant_mean=False,
                 max_val=None):
        """
        Args:
            return_alpha: IF True, return the actual alpha value instead of the alpha class. Otherwise, it returns alpha_class
            use_alpha_invariant_mean: If True, then mean and stdev corresponding to alpha=0.5 is used to normalize all inputs. If False
                                  , input is normalized with a mean,stdev computing using the alpha.
        """
        super().__init__(data_config,
                         fpath,
                         datasplit_type=datasplit_type,
                         val_fraction=val_fraction,
                         test_fraction=test_fraction,
                         normalized_input=normalized_input,
                         enable_rotation_aug=enable_rotation_aug,
                         enable_random_cropping=enable_random_cropping,
                         use_one_mu_std=use_one_mu_std,
                         allow_generation=allow_generation,
                         max_val=max_val)
        assert self._enable_random_cropping is False, "We need id for each image and so this must be false. \
            Our custom sampler will provide index with single grid_size"

        self._return_individual_channels = return_individual_channels
        self._return_alpha = return_alpha
        self._use_alpha_invariant_mean = use_alpha_invariant_mean
        print(f'[{self.__class__.__name__}] RetChannels', self._return_individual_channels, 'RetAlpha',
              self._return_alpha, 'AlphaInvMean', use_alpha_invariant_mean)

    def _compute_input(self, img_tuples, alpha_class_idx):
        if alpha_class_idx == -1:
            # alpha=0.5 is the solution.
            alpha = 0.5
        else:
            alpha, alpha_class_idx = self._sample_alpha(alpha_class_idx=alpha_class_idx)

        assert alpha is not None
        return self._compute_input_with_alpha(
            img_tuples, alpha, use_alpha_invariant_mean=self._use_alpha_invariant_mean), alpha, alpha_class_idx

    def __getitem__(self, index: Union[int, Tuple[int, int, int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, tuple) or isinstance(index, np.ndarray):
            if len(index) == 4:
                ch1_idx, ch2_idx, grid_size, alpha_class_idx = index
            elif len(index) == 3:
                ch1_idx, ch2_idx, grid_size = index
                alpha_class_idx = np.random.randint(0, high=self._ch1_alpha_interval_count) if self._is_train else -1
        else:
            ch1_idx = index
            ch2_idx = index
            grid_size = self._img_sz
            alpha_class_idx = -1

        index1 = (ch1_idx, grid_size)
        img1_tuples = self._get_img(index1)
        index2 = (ch2_idx, grid_size)
        img2_tuples = self._get_img(index2)

        assert self._enable_rotation is False
        img_tuples = (img1_tuples[0], img2_tuples[1])

        inp, alpha, _ = self._compute_input(img_tuples, alpha_class_idx=alpha_class_idx)

        alpha_val = alpha_class_idx
        if self._return_alpha:
            alpha_val = alpha

        # Filter needed in contrastive learning to ensure that zero content has its own class.
        if self._cl_std_filter is not None:
            assert len(img_tuples) == 2
            if img_tuples[0].std() <= self._cl_std_filter[0]:
                ch1_idx = -1
            if img_tuples[1].std() <= self._cl_std_filter[1]:
                ch2_idx = -1

        if self._return_individual_channels:
            target = np.concatenate(img_tuples, axis=0)
            return (inp, target, alpha_val, ch1_idx, ch2_idx)

        return inp, alpha_val, ch1_idx, ch2_idx
