"""
Here, the two images are not from same location of the same time point.
"""
from typing import Union

import numpy as np

from denoisplit.data_loader.vanilla_dloader import MultiChDloader


class MultiChDeterministicTiffRandDloader(MultiChDloader):

    def _get_img(self, index: int):
        """
        Returns the two channels. Here, for training, two randomly cropped channels are passed on.
        """
        if self._is_train:
            cropped_img1_l1, cropped_img2_l1 = super()._get_img(index)
            index = np.random.choice(np.arange(len(self)))
            cropped_img1_l2, cropped_img2_l2 = super()._get_img(index)
            if np.random.rand() > 0.5:
                return cropped_img1_l1, cropped_img2_l2
            else:
                return cropped_img1_l2, cropped_img2_l1

        else:
            # for validation, use the aligned data as this is the target.
            return super()._get_img(index)
