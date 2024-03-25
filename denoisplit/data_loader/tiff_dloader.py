import os
from typing import Tuple

import numpy as np
from skimage.io import imread
from tqdm import tqdm

from denoisplit.core.tiff_reader import load_tiff


class TiffLoader:
    def __init__(self,
                 img_sz: int,
                 enable_flips: bool = False,
                 thresh: float = None,
                 repeat_factor: int = 1,
                 normalized_input=None):
        """
        Args:
            repeat_factor: Since we are doing a random crop, repeat_factor is
            given which can repeatedly sample from the same image. If self.N=12
            and repeat_factor is 5, then index upto 12*5 = 60 is allowed.
            normalized_input: whether to normalize the input or not
        """
        assert isinstance(normalized_input, bool)
        self._img_sz = img_sz

        self._enable_flips = enable_flips
        self.N = 0
        self._avg_cropped_count = 1
        self._called_count = 0
        self._thresh = thresh
        self._repeat_factor = repeat_factor
        self._normalized_input = normalized_input
        assert self._thresh is not None

    def _crop_random(self, img1: np.ndarray, img2: np.ndarray):
        h, w = img1.shape[-2:]
        if self._img_sz is None:
            return img1, img2, {'h': [0, h], 'w': [0, w], 'hflip': False, 'wflip': False}

        h_start, w_start, h_flip, w_flip = self._get_random_hw(h, w)
        if self._enable_flips is False:
            h_flip = False
            w_flip = False

        img1 = self._crop_img(img1, h_start, w_start, h_flip, w_flip)
        img2 = self._crop_img(img2, h_start, w_start, h_flip, w_flip)

        return img1, img2, {
            'h': [h_start, h_start + self._img_sz],
            'w': [w_start, w_start + self._img_sz],
            'hflip': h_flip,
            'wflip': w_flip,
        }

    def _crop_img(self, img: np.ndarray, h_start: int, w_start: int, h_flip: bool, w_flip: bool):
        new_img = img[..., h_start:h_start + self._img_sz, w_start:w_start + self._img_sz]
        if h_flip:
            new_img = new_img[..., ::-1, :]
        if w_flip:
            new_img = new_img[..., :, ::-1]

        return new_img.astype(np.float32)

    def _get_random_hw(self, h: int, w: int):
        """
        Random starting position for the crop for the img with index `index`.
        """
        h_start = np.random.choice(h - self._img_sz)
        w_start = np.random.choice(w - self._img_sz)
        h_flip, w_flip = np.random.choice(2, size=2) == 1
        return h_start, w_start, h_flip, w_flip

    def metric(self, img: np.ndarray):
        return np.std(img)

    def in_allowed_range(self, metric_val: float):
        return metric_val >= self._thresh

    def __len__(self):
        return self.N * self._repeat_factor

    def _is_content_present(self, img1: np.ndarray, img2: np.ndarray):
        met1 = self.metric(img1)
        met2 = self.metric(img2)
        # print('Metric', met1, met2)
        if self.in_allowed_range(met1) or self.in_allowed_range(met2):
            return True
        return False

    def _load_img(self, index: int):
        """
            It must return the two images which would be mixed.
        """
        return None, None

    def _get_img(self, index: int):
        """
        Loads an image. 
        Crops the image such that cropped image has content.
        """
        img1, img2 = self._load_img(index)
        cropped_img1, cropped_img2 = self._crop_random(img1, img2)[:2]
        self._called_count += 1
        cropped_count = 1
        while (not self._is_content_present(cropped_img1, cropped_img2)):
            cropped_img1, cropped_img2 = self._crop_random(img1, img2)[:2]
            cropped_count += 1

        self._avg_cropped_count = (
            (self._called_count - 1) * self._avg_cropped_count + cropped_count) / self._called_count
        return cropped_img1, cropped_img2

    def normalize_img(self, img1, img2):
        mean, std = self.get_mean_std()
        mean = mean.squeeze()
        std = std.squeeze()
        img1 = (img1 - mean[0]) / std[0]
        img2 = (img2 - mean[1]) / std[1]
        return img1, img2

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:

        assert index < self._repeat_factor * self.N
        index = index % self.N

        img1, img2 = self._get_img(index)
        target = np.concatenate([img1, img2], axis=0)
        if self._normalized_input:
            img1, img2 = self.normalize_img(img1, img2)

        inp = (0.5 * img1 + 0.5 * img2).astype(np.float32)
        return inp, target

    def get_mean_std(self):
        return 0.0, 255.0
