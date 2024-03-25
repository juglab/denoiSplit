import numpy as np


class CropSynchronizer:
    """
    Ensures that for each noise level, same crop gets delivered.
    """
    def __init__(self, img_sz, dataset_size, max_same_crop_count, noise_levels):
        self._img_sz = img_sz
        self._size = dataset_size
        self.noise_levels = noise_levels
        # What was the last crop used.
        self._last_random_crop = None
        # How many times has the same crop being used (for each noise level).
        self._same_crop_count = None
        # number of times same crop would be used for each noise level before we randomly resample.
        self._max_same_crop_count = max_same_crop_count
        assert isinstance(self._max_same_crop_count, int)
        self.init()

    def init(self):
        self._last_random_crop = {}
        self._same_crop_count = {}
        for noise_level in self.noise_levels:
            self._same_crop_count[noise_level] = [0] * self._size
        self._last_random_crop = [None] * self._size

    def time_to_sample(self, base_index):
        if self._last_random_crop[base_index] is None:
            return True

        for noise_level in self.noise_levels:
            if self._same_crop_count[noise_level][base_index] < self._max_same_crop_count:
                return False
        return True

    def reset_crop_count(self, base_index, noise_index):
        self._same_crop_count[self.noise_levels[noise_index]][base_index] = 0

    def _increment_crop_count(self, base_index, noise_index):
        self._same_crop_count[self.noise_levels[noise_index]][base_index] += 1

    def get_hw(self, base_index, noise_index):
        self._increment_crop_count(base_index, noise_index)
        return self._last_random_crop[base_index]

    def set_hw(self, base_index, noise_index, hw):
        self._last_random_crop[base_index] = hw
        for i in range(len(self.noise_levels)):
            self.reset_crop_count(base_index, i)

        self._increment_crop_count(base_index, noise_index)

    def get_random_crop_shape(self, h, w, base_index, noise_index, force_sample=False):
        """
        Random starting position for the crop for the img with index `index`.
        """

        if force_sample is True or self.time_to_sample(base_index):
            h_start = np.random.choice(h - self._img_sz)
            w_start = np.random.choice(w - self._img_sz)
            h_flip, w_flip = np.random.choice(2, size=2) == 1
            self.set_hw(base_index, noise_index, (h_start, w_start, h_flip, w_flip))
        else:
            hw = self.get_hw(base_index, noise_index)
            h_start, w_start, h_flip, w_flip = hw

        return h_start, w_start, h_flip, w_flip
