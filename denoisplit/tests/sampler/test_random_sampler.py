import numpy as np

from denoisplit.data_loader.patch_index_manager import GridAlignement, GridIndexManager
from denoisplit.sampler.random_sampler import RandomSampler


class DummyDset:

    def __init__(self, data_shape, image_size) -> None:
        self.idx_manager = GridIndexManager(data_shape, image_size, image_size, GridAlignement.LeftTop)

    def __len__(self):
        return self.idx_manager.grid_count()


def test_default_sampler():
    """
    Tests that most indices are covered for both indices.
    Tests that grid_size is 1.
    """
    frame_size = 128
    data_shape = (30, frame_size, frame_size, 2)
    image_size = 64
    dset = DummyDset(data_shape, image_size)
    grid_size = 1
    batch_size = 32
    sampler = RandomSampler(dset, batch_size, grid_size)
    samples_per_epoch = (frame_size // image_size)**2 * data_shape[0]
    samples_per_epoch = samples_per_epoch - samples_per_epoch % batch_size

    reached_most_indices = False
    min_idx1_reached = None
    max_idx1_reached = None
    min_idx2_reached = None
    max_idx2_reached = None
    nrows = frame_size - image_size + 1
    idx_max = nrows * nrows * data_shape[0]

    for _ in range(10):
        sample_indices = []
        for batch in sampler:
            sample_indices.append(batch)
            if min_idx1_reached is None:
                idx1_values, idx2_values, grid_sizes = zip(*batch)
                assert set(grid_sizes) == {1}
                min_idx1_reached = np.min(idx1_values)
                max_idx1_reached = np.max(idx1_values)
                min_idx2_reached = np.min(idx2_values)
                max_idx2_reached = np.max(idx2_values)

        sample_indices = np.concatenate(sample_indices, axis=0)
        assert (sample_indices[:, 0] == sample_indices[:, 1]).sum() < 10
        min_idx1_reached = min(sample_indices[:, 0].min(), min_idx1_reached)
        max_idx1_reached = max(sample_indices[:, 0].max(), max_idx1_reached)
        min_idx2_reached = min(sample_indices[:, 1].min(), min_idx2_reached)
        max_idx2_reached = max(sample_indices[:, 1].max(), max_idx2_reached)

        assert len(sample_indices) == samples_per_epoch

        if max_idx1_reached - min_idx1_reached > 0.9 * idx_max and max_idx2_reached - min_idx2_reached > 0.9 * idx_max:
            reached_most_indices = True
            break
        assert reached_most_indices == True
