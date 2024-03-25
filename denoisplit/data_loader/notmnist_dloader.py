import os
import pickle
from typing import Union

import numpy as np
from skimage.io import imread
from tqdm import tqdm

from git.objects import base


class NotMNISTNoisyLoader:
    """
    """
    def __init__(self, data_fpath: str, img_files_pkl, label1, label2, return_labels: bool = False) -> None:

        # train/val split is defined in this file. It contains the list of images one needs to load from fpath_dict
        self._img_files_pkl = img_files_pkl
        self._datapath = data_fpath
        self.labels = None
        print(f'[{self.__class__.__name__}] Data fpath:', self._datapath)
        self.N = None
        self._return_labels = return_labels
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

    def _load_one_directory(self, directory, img_files_dict, labels=None):
        data_dict = {}
        if labels is None:
            labels = img_files_dict.keys()
        for label in labels:
            data = np.zeros((len(img_files_dict[label]), 27, 27), dtype=np.float32)
            for i, img_fname in tqdm(enumerate(img_files_dict[label])):
                img_fpath = os.path.join(directory, label, img_fname)
                data[i] = imread(img_fpath)

            data = np.pad(data, pad_width=((0, 0), (1, 0), (1, 0)))
            data = data[:, None, ...].copy()

            data_dict[label] = data
        return data_dict

    def load(self, labels=None):
        with open(self._img_files_pkl, 'rb') as f:
            img_files_dict = pickle.load(f)

        data = self._load_one_directory(self._datapath, img_files_dict, labels=labels)

        sz = sum([data[label].shape[0] for label in data.keys()])
        self.labels = sorted(list(data.keys()))
        label_sizes = [len(data[label]) for label in self.labels]
        self.cumlative_label_sizes = [np.sum(label_sizes[:i]) for i in range(1, 1 + len(label_sizes))]

        self.N = sz
        return data

    def __getitem__(self, index_tuple):
        index1, index2 = index_tuple
        assert index1 < self._l1_N, 'Index1 must be from first label'
        assert index2 >= self._l1_N and index2 < self.__len__(), 'Index2 must be from second label'
        img1 = self._all_data[self._l1][index1]
        img2 = self._all_data[self._l2][index2 % self._l1_N]

        inp = (img1 + img2) / 2
        target = np.concatenate([img1, img2], axis=0)
        return inp, target

    def get_mean_std(self):
        data = []
        data.append(self._all_data[self._l1])
        data.append(self._all_data[self._l2])
        all_data = np.concatenate(data)
        return np.mean(all_data), np.std(all_data)

    def __len__(self):
        return self._l1_N + self._l2_N
