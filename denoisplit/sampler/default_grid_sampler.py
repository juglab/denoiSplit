"""
The idea is one can feed the grid_size along with index.
"""
import numpy as np

from denoisplit.sampler.base_sampler import BaseSampler


class DefaultGridSampler(BaseSampler):
    """
    Randomly yields an index and an associated grid size.
    """

    def init(self):
        self.index_batches = []
        l1_idx = np.random.randint(low=0, high=self.idx_max, size=len(self._dset))
        grid_size = np.array([self._grid_size] * len(l1_idx))
        self.index_batches = list(zip(l1_idx, l1_idx, grid_size))
