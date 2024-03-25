import numpy as np
from torch.utils.data import Sampler


class BaseSampler(Sampler):
    """
    Base sampler for the class which yields wo indices
    """

    def __init__(self, dataset, batch_size, grid_size=1) -> None:
        """
        Grid size of 1 ensures that any random crop can be taken.
        """
        super().__init__(dataset)
        self._dset = dataset
        self._grid_size = grid_size
        self.idx_max = self._dset.idx_manager.grid_count(grid_size=self._grid_size)

        self._batch_size = batch_size
        self.index_batches = None
        print(f'[{self.__class__.__name__}] ')

    def init(self):
        raise NotImplementedError("This needs to be implemented")

    def __iter__(self):
        self.init()
        start_idx = 0
        for _ in range(len(self.index_batches) // self._batch_size):
            yield self.index_batches[start_idx:start_idx + self._batch_size].copy()
            start_idx += self._batch_size
