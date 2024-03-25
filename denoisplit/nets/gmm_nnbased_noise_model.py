import torch
import torch.nn as nn

from denoisplit.core.stable_exp import StableExponential
from denoisplit.nets.gmm_noise_model import GaussianMixtureNoiseModel


class PointConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, interim_channels=None, residual=False) -> None:
        super().__init__()
        if interim_channels is None:
            if in_channels < 32:
                interim_channels = 32
            else:
                interim_channels = in_channels * 2

        self.nn = nn.Sequential(
            nn.Conv2d(in_channels, interim_channels, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(interim_channels),
            nn.Conv2d(interim_channels, out_channels, 1),
            nn.LeakyReLU(),
        )
        self.residual = residual

    def forward(self, x):
        if self.residual:
            return x + self.nn(x)
        else:
            return self.nn(x)


class MuModel(nn.Module):

    def __init__(self, n_gaussian):
        super().__init__()
        self.mu_model = nn.Sequential(
            PointConvBlock(1, 32, residual=False),
            PointConvBlock(32, 32, residual=True),
            PointConvBlock(32, 32, residual=True),
            PointConvBlock(32, 32, residual=True),
            PointConvBlock(32, n_gaussian, interim_channels=32, residual=False),
        )

    def forward(self, x):
        return x + self.mu_model(x)


class DeepGMMNoiseModel(GaussianMixtureNoiseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.weight
        self.mu_model = MuModel(self.n_gaussian)

        self.sigma_model = nn.Sequential(
            PointConvBlock(1, 32, residual=False),
            PointConvBlock(32, 32, residual=True),
            PointConvBlock(32, 32, residual=True),
            PointConvBlock(32, 32, residual=True),
            PointConvBlock(32, self.n_gaussian, interim_channels=32, residual=False),
        )
        self.alpha_model = nn.Sequential(
            PointConvBlock(1, 32, residual=False),
            PointConvBlock(32, 32, residual=True),
            PointConvBlock(32, 32, residual=True),
            PointConvBlock(32, 32, residual=True),
            PointConvBlock(32, self.n_gaussian, interim_channels=32, residual=False),
        )

    def make_learnable(self):
        print(f'[{self.__class__.__name__}] Making noise model learnable')
        self._learnable = True
        # for params in self.parameters():
        #     params.requires_grad = True

    def to_device(self, cuda_tensor):
        if self.min_signal.device != cuda_tensor.device:
            self.max_signal = self.max_signal.to(cuda_tensor.device)
            self.min_signal = self.min_signal.to(cuda_tensor.device)
            self.tol = self.tol.to(cuda_tensor.device)

    def getGaussianParameters(self, signals):
        """Returns the noise model for given signals
                Parameters
                ----------
                signals : torch.cuda.FloatTensor
                    Underlying signals
                Returns
                -------
                noiseModel: list of torch.cuda.FloatTensor
                    Contains a list of `mu`, `sigma` and `alpha` for the `signals`
        """
        noiseModel = []
        mu = []
        sigma = []
        alpha = []
        mu = [self.mu_model(signals)[:, k:k + 1] for k in range(self.n_gaussian)]

        sigmaTemp = StableExponential(self.sigma_model(signals)).exp()
        sigmaTemp = torch.clamp(sigmaTemp, min=self.min_sigma)
        sigmaTemp = torch.sqrt(sigmaTemp)
        sigma = [sigmaTemp[:, k:k + 1] for k in range(self.n_gaussian)]
        alphatemp = StableExponential(self.alpha_model(signals)).exp() + self.tol
        alpha = [alphatemp[:, k:k + 1] for k in range(self.n_gaussian)]

        sum_alpha = 0
        for al in range(self.n_gaussian):
            sum_alpha = alpha[al] + sum_alpha
        for ker in range(self.n_gaussian):
            alpha[ker] = alpha[ker] / sum_alpha

        sum_means = 0
        for ker in range(self.n_gaussian):
            sum_means = alpha[ker] * mu[ker] + sum_means

        mu_shifted = []
        for ker in range(self.n_gaussian):
            mu[ker] = mu[ker] - sum_means + signals

        for i in range(self.n_gaussian):
            noiseModel.append(mu[i])
        for j in range(self.n_gaussian):
            noiseModel.append(sigma[j])
        for k in range(self.n_gaussian):
            noiseModel.append(alpha[k])

        return noiseModel
