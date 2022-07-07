import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# custom modules
from .LocalLearning import KHL3

def stringer_spectrum(nu):
    return 1 / nu

def cov_spectrum(
        dataloader: DataLoader, model: KHL3, device: torch.device
) -> Tensor:
    """
    Calculate the ordered spectrum of the covariance matrix
    """
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as ttest:

            noE = model.pSet["hidden_size"]

            """
            Calculate covariance matrix on shifted data to increase
            computational stability
            """

            # initialize stat tensors
            corr = torch.zeros((noE, noE), device=device)
            mean = torch.zeros((noE,), device=device)
            pop_size = 0

            ttest.set_description(f"Calculating covariance spectrum")

            # shift = estimate of the mean
            f, label = next(iter(dataloader))
            K = f.mean()

            for x, label in ttest:
                pop_size += len(x)
                pred = model(x.to(device))
                corr += (pred.T - K) @ (pred - K)
                mean += pred.sum(axis=0) - K

            cov = corr - mean.T @ mean / pop_size
            cov /= pop_size - 1

            l_real = torch.linalg.eigvals(cov).real
            l_sorted, idx = torch.sort(l_real, descending=True)
            return l_sorted

