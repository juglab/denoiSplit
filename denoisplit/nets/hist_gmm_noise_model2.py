import math

import numpy as np
import torch
from scipy.optimize import curve_fit


def gaus(x, mu, sigma):
    out = np.exp(-(x - mu)**2 / (2 * sigma**2)) * 1 / (sigma * np.sqrt(2 * math.pi))
    return out


def gaus_pytorch(x, mu, sigma):
    out = torch.exp(-(x - mu)**2 / (2 * sigma**2)) * 1 / (sigma * np.sqrt(2 * math.pi))
    return out


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class HistGMMNoiseModel:

    def __init__(self, histdata) -> None:
        self._histdata = histdata
        bin_val = (self._histdata[1] + self._histdata[2]) / 2
        # midpoint of every bin
        self._bin_val = bin_val[:, 0]
        self._binsize = np.mean(self._histdata[2] - self._histdata[1])
        # probability density function.
        self._bin_pdf = self._histdata[0] / self._binsize
        self._params = []

        self.minv = np.min(histdata[1, ...])

        # The upper boundaries of each bin in y are stored in dimension 2
        self.maxv = np.max(histdata[2, ...])
        self.bins = histdata.shape[1]
        self._min_valid_index = None
        self._max_valid_index = None
        self.tol = 1e-10

    def fit_index(self, index):
        x = self._bin_val
        y = self._bin_pdf[index]
        if y.sum() * self._binsize < 1e-5:
            return torch.tensor([torch.nan, torch.nan])

        if self._min_valid_index is not None:
            self._min_valid_index = min(index, self._min_valid_index)
        else:
            self._min_valid_index = index

        if self._max_valid_index is not None:
            self._max_valid_index = max(index, self._max_valid_index)
        else:
            self._max_valid_index = index

        assert abs(y.sum() * self._binsize - 1) < 1e-5

        mean = self._bin_val[index]
        sigma = sum(y * (x - mean)**2)
        popt, pcov = curve_fit(gaus, x, y, p0=[x[index], sigma], maxfev=6000)
        return torch.Tensor(popt)

    def fit(self):
        for index in range(len(self._bin_pdf)):
            popt = self.fit_index(index)
            self._params.append(popt)

        self._params = torch.stack(self._params)
        # manually adde after last and before first bin.
        if self._min_valid_index > 0:
            self._params[self._min_valid_index - 1] = self._params[self._min_valid_index]
            self._params[self._min_valid_index - 1, 0] -= self._binsize
            self._min_valid_index -= 1

        if self._max_valid_index < self.bins - 1:
            self._params[self._max_valid_index + 1] = self._params[self._max_valid_index]
            self._params[self._max_valid_index + 1, 0] += self._binsize
            self._max_valid_index += 1

        self._params = self._params.cuda()

    def getIndexSignalFloat(self, x):
        return torch.clamp(self.bins * (x - self.minv) / (self.maxv - self.minv), min=0.0, max=self.bins - 1 - 1e-3)

    def likelihood(self, obs, signal):
        signalF = self.getIndexSignalFloat(signal)
        signal_ = signalF.floor().long()
        fact = signalF - signal_.float()
        underflow_mask = signal_ < self._min_valid_index
        signal_[underflow_mask] = self._min_valid_index
        fact[underflow_mask] = 0.0

        overflow_mask = signal_ > self._max_valid_index
        signal_[overflow_mask] = self._max_valid_index
        params1 = self._params[signal_]
        mu1 = params1[..., 0]
        sigma1 = params1[..., 1]

        # if the signal is in the last bin, we just need to ignore the first mu and sigma and go with the last one.
        last_index_mask = signal_ == self._max_valid_index
        signal_[last_index_mask] = self._max_valid_index - 1
        fact[last_index_mask] = 1.0

        params2 = self._params[signal_ + 1]
        mu2 = params2[..., 0]
        sigma2 = params2[..., 1]
        mu = mu1 * (1 - fact) + mu2 * fact
        sigma = sigma1 * (1 - fact) + sigma2 * fact
        return self.tol + gaus_pytorch(obs, mu, sigma)
