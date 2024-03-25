from typing import Tuple, Union

import numpy as np

from denoisplit.data_loader.lc_multich_dloader import LCMultiChDloader


class LCMultiChExplicitInputDloader(LCMultiChDloader):
    """
    The first index of the data is the input, other indices are targets.
    # 1. mean, stdev needs to handled differently for input and target.
    # 2. input computation will ofcourse be different.
    Note that for normalizing the input, we compute the stats from all the channels of the data. One might want to 
    compute the stats from the first channel only. 
    """

    def get_mean_std_for_input(self):
        mean, std = super().get_mean_std_for_input()
        return mean[:, :1], std[:, :1]

    def compute_individual_mean_std(self):
        """
        Here, we remove the mean and stdev computation for the input.
        """
        mean, std = super().compute_individual_mean_std()
        return mean[:, 1:], std[:, 1:]

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        img_tuples, noise_tuples = self._get_img(index)
        assert self._enable_rotation is False
        assert len(noise_tuples) == 0, 'Noise is not supported in this data loader.'
        assert self._lowres_supervision != True
        target = np.concatenate([img[:1] for img in img_tuples[1:]], axis=0)
        input_tuples = img_tuples[:1]
        inp, alpha = self._compute_input(input_tuples)

        output = [inp, target]

        if self._return_alpha:
            output.append(alpha)

        if isinstance(index, int):
            return tuple(output)

        _, grid_size = index
        output.append(grid_size)
        return tuple(output)
