"""
In this sampler, we want to make sure that if one patch goes into the batch,
its four neighbors also go in the same patch. 
A batch is an ordered sequence of inputs in groups of 5.
An example batch of size 16:
A1,A2,A3,A4,A5, B1,B2,B3,B4,B5, C1,C2,C3,C4,C5, D1

First element (A1) is the central element.
2nd (A2), 3rd(A3), 4th(A4), 5th(A5) elements are left, right, top, bottom
"""
import numpy as np
from torch.utils.data import Sampler


class BaseSampler(Sampler):
    def __init__(self, dataset, batch_size) -> None:
        super().__init__(dataset)
        self._dset = dataset
        self._batch_size = batch_size
        self.idx_manager = self._dset.idx_manager
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


class NeighborSampler(BaseSampler):
    def __init__(self, dataset, batch_size, nbr_set_count=None, valid_gridsizes=None) -> None:
        """
        Args:
            nbr_set_count: how many set of neighbors should be provided. They are present in the beginning of the batch. 
                        nbr_set_count=2 will mean 2 sets of neighbors are provided in each batch. And they will comprise first 10 instances in the batch.
                        Remaining elements in the batch will be drawn randomly.
        """
        super().__init__(dataset, batch_size)
        self._valid_gridsizes = valid_gridsizes
        self._nbr_set_count = nbr_set_count
        print(f'[{self.__class__.__name__}] NbrSet:{self._nbr_set_count}')

    def dset_len(self, grid_size):
        return self.idx_manager.grid_count(grid_size=grid_size)

    def _add_one_batch(self):
        rand_sz = int(np.ceil(self._batch_size / 5))
        if self._nbr_set_count is not None:
            rand_sz = min(rand_sz, self._nbr_set_count)

        rand_idx_list = []
        rand_grid_list = []
        for _ in range(rand_sz):
            grid_size = np.random.choice(self._valid_gridsizes) if self._valid_gridsizes is not None else 1
            rand_grid_list.append(grid_size)
            idx = np.random.randint(self.dset_len(grid_size))
            while self.idx_manager.on_boundary(idx, grid_size=grid_size):
                idx = np.random.randint(self.dset_len(grid_size))
            rand_idx_list.append(idx)

        batch_idx_list = []
        for rand_idx, grid_size in zip(rand_idx_list, rand_grid_list):
            batch_idx_list.append((rand_idx, grid_size))
            batch_idx_list.append((self.idx_manager.get_left_nbr_idx(rand_idx, grid_size=grid_size), grid_size))
            batch_idx_list.append((self.idx_manager.get_right_nbr_idx(rand_idx, grid_size=grid_size), grid_size))
            batch_idx_list.append((self.idx_manager.get_top_nbr_idx(rand_idx, grid_size=grid_size), grid_size))
            batch_idx_list.append((self.idx_manager.get_bottom_nbr_idx(rand_idx, grid_size=grid_size), grid_size))

        if self._nbr_set_count is not None and len(batch_idx_list) < self._batch_size:
            grid_size = 1  # This size ensures that patch can begin at any random pixel.
            idx_list = list(np.random.randint(self.dset_len(grid_size), size=self._batch_size - len(batch_idx_list)))
            gridsizes = [grid_size] * len(idx_list)
            batch_idx_list += zip(idx_list, gridsizes)
            self.index_batches += batch_idx_list
        else:
            self.index_batches += batch_idx_list[:self._batch_size]

    def init(self):
        self.index_batches = []
        num_batches = len(self._dset) // self._batch_size
        for _ in range(num_batches):
            self._add_one_batch()
