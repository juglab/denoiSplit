"""
Multi channel deterministic tiff data loader which takes as input two indices: one for each channel
"""

import numpy as np

from denoisplit.data_loader.vanilla_dloader import MultiChDloader


class TwinIndexDloader(MultiChDloader):

    def __getitem__(self, idx):
        idx1, idx2 = idx
        img1, _ = self._get_img(idx1)
        _, img2 = self._get_img(idx2)

        if self._enable_rotation:
            rot_dic = self._rotation_transform(image=img1[0], mask=img2[0])
            img1 = rot_dic['image'][None]
            img2 = rot_dic['mask'][None]
        target = np.concatenate([img1, img2], axis=0)
        if self._normalized_input:
            img1, img2 = self.normalize_img(img1, img2)

        inp = (0.5 * img1 + 0.5 * img2).astype(np.float32)
        return inp, target
