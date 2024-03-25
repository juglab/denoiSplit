from typing import Tuple

import numpy as np

from denoisplit.core.tiff_reader import load_tiff
from denoisplit.data_loader.tiff_dloader import TiffLoader


class CngbMitoActinLoader(TiffLoader):
    def __init__(self,
                 img_sz: int,
                 mito_fpath: str,
                 actin_fpath: str,
                 enable_flips: bool = False,
                 thresh: float = None):
        super().__init__(img_sz, enable_flips=enable_flips, thresh=thresh)
        self._mito_fpath = mito_fpath
        self._actin_fpath = actin_fpath

        self._mito_data = load_tiff(self._mito_fpath).astype(np.float32)
        fac = 255 / self._mito_data.max()
        self._mito_data *= fac

        self._actin_data = load_tiff(self._actin_fpath).astype(np.float32)
        fac = 255 / self._actin_data.max()
        self._actin_data *= fac

        assert len(self._mito_data) == len(self._actin_data)
        self.N = len(self._mito_data)

    def _load_img(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        img1 = self._mito_data[index]
        img2 = self._actin_data[index]
        return img1[None], img2[None]
