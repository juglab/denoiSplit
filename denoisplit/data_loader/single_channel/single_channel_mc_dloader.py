"""
Here, the input image is of multiple resolutions. Target image is the same.
"""
from typing import List, Tuple, Union

import numpy as np
from skimage.transform import resize
from denoisplit.core.data_split_type import DataSplitType

from denoisplit.data_loader.single_channel.single_channel_dloader import SingleChannelDloader
from denoisplit.core.data_type import DataType


class SingleChannelMSDloader(SingleChannelDloader):

    def __init__(
        self,
        data_config,
        fpath: str,
        datasplit_type: DataSplitType = None,
        val_fraction=None,
        test_fraction=None,
        normalized_input=None,
        enable_rotation_aug: bool = False,
        use_one_mu_std=None,
        num_scales: int = None,
        enable_random_cropping=False,
        padding_kwargs: dict = None,
        allow_generation: bool = False,
        max_val=None,
    ):
        """
        Args:
            num_scales: The number of resolutions at which we want the input. Note that the target is formed at the
                        highest resolution.
        """
        self._padding_kwargs = padding_kwargs  # mode=padding_mode, constant_values=constant_value
        super().__init__(data_config,
                         fpath,
                         datasplit_type=datasplit_type,
                         val_fraction=val_fraction,
                         test_fraction=test_fraction,
                         normalized_input=normalized_input,
                         enable_rotation_aug=enable_rotation_aug,
                         enable_random_cropping=enable_random_cropping,
                         use_one_mu_std=use_one_mu_std,
                         allow_generation=allow_generation,
                         max_val=max_val)

        self.num_scales = num_scales
        assert self.num_scales is not None
        self._scaled_data = [self._data]
        assert isinstance(self.num_scales, int) and self.num_scales >= 1
        # self.enable_padding_while_cropping is used only for overlapping_dloader. This is a hack and at some point be
        # fixed properly
        self.enable_padding_while_cropping = False
        assert isinstance(self._padding_kwargs, dict)
        assert 'mode' in self._padding_kwargs

        for _ in range(1, self.num_scales):
            shape = self._scaled_data[-1].shape
            assert len(shape) == 4
            new_shape = (shape[0], shape[1] // 2, shape[2] // 2, shape[3])
            ds_data = resize(self._scaled_data[-1], new_shape)
            self._scaled_data.append(ds_data)

    def _init_msg(self):
        msg = super()._init_msg()
        msg += f' Pad:{self._padding_kwargs}'
        return msg

    def _load_scaled_img(self, scaled_index, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            idx = index
        else:
            idx, _ = index
        imgs = self._scaled_data[scaled_index][idx % self.N]
        return imgs[None, :, :, 0], imgs[None, :, :, 1]

    def _crop_img(self, img: np.ndarray, h_start: int, w_start: int):
        """
        Here, h_start, w_start could be negative. That simply means we need to pick the content from 0. So,
        the cropped image will be smaller than self._img_sz * self._img_sz
        """
        h_end = h_start + self._img_sz
        w_end = w_start + self._img_sz
        h_start = max(0, h_start)
        w_start = max(0, w_start)
        new_img = img[..., h_start:h_end, w_start:w_end]
        return new_img

    def _get_img(self, index: int):
        """
        Loads an image.
        Crops the image such that cropped image has content.
        """
        img1, img2 = self._load_img(index)
        assert self._img_sz is not None
        h, w = img1.shape[-2:]
        if self._enable_random_cropping:
            h_start, w_start = self._get_random_hw(h, w)
        else:
            h_start, w_start = self._get_deterministic_hw(index)
        img1_cropped = self._crop_flip_img(img1, h_start, w_start, False, False)
        img2_cropped = self._crop_flip_img(img2, h_start, w_start, False, False)

        h_center = h_start + self._img_sz // 2
        w_center = w_start + self._img_sz // 2
        img1_versions = [img1_cropped]
        img2_versions = [img2_cropped]
        for scale_idx in range(1, self.num_scales):
            img1, img2 = self._load_scaled_img(scale_idx, index)
            h_center = h_center // 2
            w_center = w_center // 2
            h_start = h_center - self._img_sz // 2
            w_start = w_center - self._img_sz // 2

            img1_cropped = self._crop_flip_img(img1, h_start, w_start, False, False)
            img2_cropped = self._crop_flip_img(img2, h_start, w_start, False, False)

            h_start = max(0, -h_start)
            w_start = max(0, -w_start)
            h_end = h_start + img1_cropped.shape[1]
            w_end = w_start + img1_cropped.shape[2]
            if self.enable_padding_while_cropping:
                assert img1_cropped.shape == img1_versions[-1].shape
                assert img2_cropped.shape == img2_versions[-1].shape
                img1_padded = img1_cropped
                img2_padded = img2_cropped

            else:
                h_max, w_max = img1_versions[-1].shape[1:]
                assert img1_versions[-1].shape == img2_versions[-1].shape
                padding = np.array([[0, 0], [h_start, h_max - h_end], [w_start, w_max - w_end]])
                # mode=padding_mode, constant_values=constant_value
                img1_padded = np.pad(img1_cropped, padding, **self._padding_kwargs)
                img2_padded = np.pad(img2_cropped, padding, **self._padding_kwargs)

                # img1_padded[:, h_start:h_end, w_start:w_end] = img1_cropped
                # img2_padded[:, h_start:h_end, w_start:w_end] = img2_cropped

            img1_versions.append(img1_padded)
            img2_versions.append(img2_padded)

        img1 = np.concatenate(img1_versions, axis=0)
        img2 = np.concatenate(img2_versions, axis=0)
        return img1, img2

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        inp, target = self._get_img(index)
        target = target[:1]  # we don't need lower resolution for target.
        assert self._enable_rotation is False
        inp = self.normalize_input(inp)

        if isinstance(index, int):
            return inp, target
        _, grid_size = index
        return inp, target, grid_size
