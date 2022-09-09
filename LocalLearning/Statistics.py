import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# custom modules
from .LocalLearning import KHL3


def stringer_spectrum(nu):
    return 1 / nu


def cov_spectrum(
        dataloader: DataLoader, model: Module, device: torch.device, dtype: torch.dtype=torch.float32
) -> Tensor:
    """
    Calculate the ordered singular value spectrum of the covariance matrix
    """
    with torch.no_grad():

        noE = model.pSet["hidden_size"]

        """
        Calculate covariance matrix on shifted data to increase
        computational stability
        """

        # initialize stat tensors
        auto_corr = torch.zeros((noE, noE), device=device, dtype=dtype)
        mean = torch.zeros((noE,), device=device, dtype=dtype)
        pop_size = 0

        # shift = estimate of the mean
        f, _ = next(iter(dataloader))
        K = f.mean()

        with tqdm(dataloader, unit="batch") as ttest:
            ttest.set_description(f"Calculating covariance spectrum")
            for x, _ in ttest:
                pop_size += len(x)
                pred = model(x.to(device))
                data = pred - K
                auto_corr += data.T @ data
                mean += data.sum(axis=0)

        cov = auto_corr - mean[None].T @ mean[None] / pop_size
        cov /= pop_size - 1

        # cov matrix is real valued
        # -> [cov, cov.adjoint] = 0
        # -> eigen values are real
        l_real = torch.linalg.eigvals(cov).real
        l_n, _ = torch.sort(l_real, descending=True)

        return l_n
