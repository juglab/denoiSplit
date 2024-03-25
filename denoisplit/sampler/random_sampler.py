import numpy as np

from denoisplit.sampler.base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    """
    Randomly yields the two indices
    """

    def init(self):
        self.index_batches = []
        l1_idx = np.random.randint(low=0, high=self.idx_max, size=len(self._dset))
        l2_idx = np.random.randint(low=0, high=self.idx_max, size=len(self._dset))
        grid_size = np.array([self._grid_size] * len(l2_idx))
        self.index_batches = list(zip(l1_idx, l2_idx, grid_size))
