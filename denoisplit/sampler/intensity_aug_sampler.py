import numpy as np
from torch.utils.data import Sampler


class LevelIndexIterator:

    def __init__(self, index_list) -> None:
        self._index_list = index_list
        self._N = len(self._index_list)
        self._cur_position = 0

    def next(self):
        output_pos = self._cur_position
        self._cur_position += 1
        self._cur_position = self._cur_position % self._N
        return self._index_list[output_pos]

    def next_k(self, N):
        return [self.next() for _ in range(N)]


class IntensityAugValSampler(Sampler):
    INVALID = -955

    def __init__(self, dataset, grid_size, batch_size, fixed_alpha_idx=-1) -> None:
        super().__init__(dataset)
        # In validation, we just look at the cases which we'll find in the test case. alpha=0.5 is that case. This corresponds to the -1 class.
        self._alpha_idx = fixed_alpha_idx
        self._N = len(dataset)
        self._batch_N = batch_size
        self._grid_size = grid_size

    def __iter__(self):
        num_batches = int(np.ceil(self._N / self._batch_N))
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self._batch_N
            end_idx = min((batch_idx + 1) * self._batch_N, self._N)
            # 4 channels: ch1_idx, ch2_idx, grid_size, alpha_idx
            batch_data_idx = np.ones((end_idx - start_idx, 4), dtype=np.int32) * self.INVALID
            batch_data_idx[:, 0] = np.arange(start_idx, end_idx)
            batch_data_idx[:, 1] = batch_data_idx[:, 0]
            batch_data_idx[:, 2] = self._grid_size
            batch_data_idx[:, 3] = self._alpha_idx
            yield batch_data_idx


class IntensityAugSampler(Sampler):
    INVALID = -955

    def __init__(self,
                 dataset,
                 data_size,
                 ch1_alpha_interval_count,
                 num_intensity_variations,
                 batch_size,
                 fixed_alpha=None) -> None:
        super().__init__(dataset)
        self._dset = dataset
        self._N = data_size
        self._alpha_class_N = ch1_alpha_interval_count
        self._fixed_alpha = fixed_alpha
        self._batch_N = batch_size
        self._intensity_N = num_intensity_variations
        assert batch_size % self._intensity_N == 0
        # We'll be using grid_size of 1, this allows us to pick from any random location in the frame. However,
        # as far as one epoch is concerned, we'll use data_size. So, values in self.idx will be much larger than
        # self._N
        self._grid_size = 1
        self.idx = np.arange(self._dset.idx_manager.grid_count(grid_size=self._grid_size))
        self.batches_idx_list = None
        self.level_iters = None
        print(f'[{self.__class__.__name__}] Alpha class count:{self._alpha_class_N}')

    def __iter__(self):
        """
        Here, we make sure that self._intensity_N many intensity variations of the same two channels are fed
        as input.
        """
        self.init()
        for one_batch_idx in self.batches_idx_list:
            alpha_idx_list, idx_list = one_batch_idx

            # 4 channels: ch1_idx, ch2_idx, grid_size, alpha_idx
            batch_data_idx = np.ones((self._batch_N, 4), dtype=np.int32) * self.INVALID
            # grid size will always be 1.
            batch_data_idx[:, 0] = idx_list
            batch_data_idx[:, 1] = idx_list
            batch_data_idx[:, 2] = self._grid_size
            batch_data_idx[:, 3] = alpha_idx_list

            assert (batch_data_idx == self.INVALID).any() == False
            yield batch_data_idx

    def init(self):
        self.batches_idx_list = []
        total_size = self._N
        num_batches = int(np.ceil(total_size / self._batch_N))
        idx = self.idx.copy()
        np.random.shuffle(idx)
        self.idx_iterator = LevelIndexIterator(idx)

        idx = self.idx.copy()
        np.random.shuffle(idx)

        for _ in range(num_batches):
            idx_list = self.idx_iterator.next_k(self._batch_N // self._intensity_N)
            alpha_list = []
            for _ in idx_list:
                if self._fixed_alpha:
                    alpha_idx = np.array([-1] * self._alpha_class_N)
                else:
                    alpha_idx = np.random.choice(np.arange(self._alpha_class_N), size=self._intensity_N, replace=False)
                alpha_list.append(alpha_idx)

            alpha_list = np.concatenate(alpha_list)
            idx_list = np.tile(np.array(idx_list).reshape(-1, 1), (1, self._intensity_N)).reshape(-1)
            self.batches_idx_list.append((alpha_list, idx_list))


if __name__ == '__main__':
    from denoisplit.data_loader.patch_index_manager import GridAlignement, GridIndexManager
    grid_size = 1
    patch_size = 64
    grid_alignment = GridAlignement.LeftTop

    class DummyDset:

        def __init__(self) -> None:
            self.idx_manager = GridIndexManager((6, 2400, 2400, 2), grid_size, patch_size, grid_alignment)

    ch1_alpha_interval_count = 30
    data_size = 1000
    num_intensity_variations = 2
    batch_size = 32
    sampler = IntensityAugSampler(DummyDset(), data_size, ch1_alpha_interval_count, num_intensity_variations,
                                  batch_size)
    for batch in sampler:
        break

    print('')
