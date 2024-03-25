from denoisplit.sampler.twin_index_sampler import TwinIndexSampler
import numpy as np


def test_twin_index_sampler():
    """
    Test makes only sense if the size of the dataset is a multiple of the batch_size
    """
    batch_size = 12

    class DummyDataset:
        def __len__(self):
            return batch_size * 5

        def __getitem__(self, index):
            idx1, idx2 = index
            return np.random.rand(4, 4), np.random.rand(4, 4)

    dset = DummyDataset()
    sampler = TwinIndexSampler(dset, batch_size)

    all_tuples = []
    for batch_idx in sampler:
        all_tuples += batch_idx
    a, b = zip(*all_tuples)
    assert set(a) == set(b)
    assert len(a) == 2 * len(set(a))
    assert max(a) == len(dset) - 1
    assert min(a) == 0
    assert sum(a) == (len(dset) - 1) * len(dset)
