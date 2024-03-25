from torch.utils.data import Sampler
import numpy as np
from torch.utils.data import Sampler


class TwinIndexSampler(Sampler):
    """
    This indexer returns a tuple index instead of an integer index.
    So, if batch size is 4, then something like this is returned for one batch:
        [(0,4), (0,5), (31,4), (31,5)]
    """

    def __init__(self, dataset, batch_size) -> None:
        super().__init__(dataset)
        self._dset = dataset
        self._N = len(self._dset)

        self._batch_size = batch_size
        assert batch_size % 4 == 0
        self.index_batches = None

    def __iter__(self):
        self.init()
        for one_batch_idx in self.index_batches:
            yield one_batch_idx

    def all_combinations(self, l1, l2):
        """
        Returns an array with 4 tuples: every combination of l1 and l2.
        """
        assert len(l1) == 2
        assert len(l2) == 2
        return [
            (l1[0], l2[0]),
            (l1[0], l2[1]),
            (l1[1], l2[0]),
            (l1[1], l2[1])]

    def _get_batch_idx_tuples(self, label1_indices, label2_indices):
        batch_indices = []
        assert len(label1_indices) % 2 == 0

        for i in range(len(label1_indices) // 2):
            batch_indices += self.all_combinations(label1_indices[2 * i:2 * i + 2],
                                                   label2_indices[2 * i:2 * i + 2])
        return batch_indices

    def init(self):
        self.index_batches = []
        uniq_idx_batch = self._batch_size // 2

        data1_idx = np.arange(self._N)
        np.random.shuffle(data1_idx)

        data2_idx = np.arange(self._N)
        np.random.shuffle(data2_idx)

        for batch_num in range((self._N // uniq_idx_batch)):
            start = uniq_idx_batch * batch_num
            end = start + uniq_idx_batch
            if end > self._N:
                break

            batch_idx = self._get_batch_idx_tuples(data1_idx[start:end], data2_idx[start:end])
            self.index_batches.append(batch_idx)
