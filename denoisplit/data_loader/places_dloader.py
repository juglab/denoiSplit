import os
import pickle
from typing import Union

import numpy as np
from skimage.io import imread
from tqdm import tqdm


class PlacesLoader:
    """
    """
    def __init__(self, data_fpath: str, label1, label2, return_labels: bool = False, img_dsample=None) -> None:

        self._datapath = data_fpath
        self.labels = None
        print(f'[{self.__class__.__name__}] Data fpath:', self._datapath, f'{label1} {label2}')
        self.N = None
        self._return_labels = return_labels
        self._img_dsample = img_dsample
        self._l1 = label1
        self._l2 = label2
        self._all_data = self.load(labels=[self._l1, self._l2])
        self._l1_index = self.labels.index(label1)
        self._l2_index = self.labels.index(label2)
        self._l1_N = len(self._all_data[label1])
        self._l2_N = len(self._all_data[label2])

    def get_label_idx_range(self):
        return {
            '1': [0, self._l1_N],
            '2': [self._l1_N, self._l1_N + self._l2_N],
        }

    def _load_label(self, directory, label):
        label_direc = os.path.join(directory, label)
        fpaths = []
        for img_fname in os.listdir(label_direc):
            img_fpath = os.path.join(label_direc, img_fname)
            fpaths.append(img_fpath)

        return sorted(fpaths)

    def _load(self, directory, labels):
        data_dict = {}
        for label in labels:
            data = self._load_label(directory, label)
            data_dict[label] = data
        return data_dict

    def load(self, labels=None):
        data = self._load(self._datapath, labels=labels)

        sz = sum([len(data[label]) for label in data.keys()])
        self.labels = sorted(list(data.keys()))
        label_sizes = [len(data[label]) for label in self.labels]
        self.cumlative_label_sizes = [np.sum(label_sizes[:i]) for i in range(1, 1 + len(label_sizes))]

        self.N = sz
        return data

    def _get_img(self, img_fpath):
        img = imread(img_fpath)
        # downsampling the image.
        img = img[::self._img_dsample, ::self._img_dsample]
        # img = np.pad(img, pad_width=((1, 0), (1, 0)))
        img = img[None]
        return img

    def __getitem__(self, index_tuple):
        index1, index2 = index_tuple
        assert index1 < self._l1_N, 'Index1 must be from first label'
        assert index2 >= self._l1_N and index2 < self.__len__(), 'Index2 must be from second label'
        img1 = self._get_img(self._all_data[self._l1][index1])
        img2 = self._get_img(self._all_data[self._l2][index2 % self._l1_N])

        inp = (0.5 * img1 + 0.5 * img2).astype(np.float32)
        target = np.concatenate([img1, img2], axis=0)
        return inp, target

    def get_mean_std(self):
        return 0.0, 255.0

    def __len__(self):
        return self._l1_N + self._l2_N
