import numpy as np

from denoisplit.data_loader.patch_index_manager import GridAlignement, GridIndexManager
from denoisplit.sampler.default_grid_sampler import DefaultGridSampler


class DummyDset:

    def __init__(self, data_shape, image_size) -> None:
        self.idx_manager = GridIndexManager(data_shape, image_size, image_size, GridAlignement.LeftTop)

    def __len__(self):
        return self.idx_manager.grid_count()


def test_default_sampler():
    """
    Tests that most indices are covered.
    Tests that grid_size is 1.
    """
    frame_size = 128
    data_shape = (30, frame_size, frame_size, 2)
    image_size = 64
    dset = DummyDset(data_shape, image_size)
    grid_size = 1
    batch_size = 32
    sampler = DefaultGridSampler(dset, batch_size, grid_size)
    samples_per_epoch = (frame_size // image_size)**2 * data_shape[0]
    samples_per_epoch = samples_per_epoch - samples_per_epoch % batch_size

    reached_most_indices = False
    min_idx_reached = None
    max_idx_reached = None
    nrows = frame_size - image_size + 1
    idx_max = nrows * nrows * data_shape[0]

    for _ in range(10):
        sample_indices = []
        for batch in sampler:
            sample_indices.append(batch)
            if min_idx_reached is None:
                idx_values, same_idx_values, grid_sizes = zip(*batch)
                assert set(grid_sizes) == {1}
                assert np.all(same_idx_values == idx_values)

                min_idx_reached = np.min(idx_values)
                max_idx_reached = np.max(idx_values)

        sample_indices = np.concatenate(sample_indices, axis=0)
        min_idx_reached = min(sample_indices[:, 0].min(), min_idx_reached)
        max_idx_reached = max(sample_indices[:, 0].max(), max_idx_reached)
        assert len(sample_indices) == samples_per_epoch

        if max_idx_reached - min_idx_reached > 0.9 * idx_max:
            reached_most_indices = True
            break
        assert reached_most_indices == True
