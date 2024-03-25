import numpy as np
from torch.utils.data import Sampler

from denoisplit.sampler.base_sampler import BaseSampler


class SingleImgSampler(BaseSampler):
    """
    Ensures that in one batch, one image is same across the batch. other image changes.
    """
    def init(self):
        self.index_batches = []

        l1_range = self.label_idx_dict['1']
        l2_range = self.label_idx_dict['2']
        N = self._batch_size

        num_batches = len(self._dset) // N
        # In half of the batches label1 image will be same. In the other half label2 image will be the same
        # SI ~ single image
        SI_cnt = int(np.ceil(num_batches / 2))

        l1_SI_idx = np.random.choice(np.arange(l1_range[0], l1_range[1]), size=SI_cnt, replace=SI_cnt > self.l1_N)
        l2_SI_idx = np.random.choice(np.arange(l2_range[0], l2_range[1]), size=SI_cnt, replace=SI_cnt > self.l2_N)

        l1_idx = np.random.choice(np.arange(l1_range[0], l1_range[1]), size=SI_cnt * N, replace=SI_cnt * N > self.l1_N)
        l2_idx = np.random.choice(np.arange(l2_range[0], l2_range[1]), size=SI_cnt * N, replace=SI_cnt * N > self.l2_N)
        for i in range(num_batches):
            iby2 = i // 2
            if i % 2 == 0:
                self.index_batches += list(zip([l1_SI_idx[iby2]] * N, l2_idx[iby2 * N:(iby2 + 1) * N]))
            else:
                self.index_batches += list(zip(l1_idx[iby2 * N:(iby2 + 1) * N], [l2_SI_idx[iby2]] * N))
