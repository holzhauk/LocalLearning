import torch
from torch import Tensor
from LocalLearning import HiddenLayerModel
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

# custom modules
from .LocalLearning import KHL3


def stringer_spectrum(nu):
    return 1 / nu


def cov_spectrum(
        dataloader: DataLoader, 
        model: torch.nn.Module, 
        no_hidden_elements: int = None,
        dtype: torch.dtype=torch.float32,
) -> Tensor:
    """
    Calculate the ordered singular value spectrum of the covariance matrix

    """
    
    def _get_hidden_other(model, x:torch.Tensor) -> torch.Tensor:
        return model(x.to(device))

    def _get_hidden_HLM(model, x:torch.Tensor) -> torch.Tensor:
        _, hidden = model(x.to(device))
        return hidden

    def get_hidden(model, x: torch.Tensor) -> torch.Tensor:
        pass

    if issubclass(type(model), HiddenLayerModel):
        get_hidden = _get_hidden_HLM
        device = model.device
    else:
        get_hidden = _get_hidden_other
        p0 = next(iter(model.parameters()))
        device = p0.device
    
    with torch.no_grad():
        noE = 0
        if no_hidden_elements is None:
            noE = model.pSet["hidden_size"]
        else:
            noE = no_hidden_elements

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
                hidden = get_hidden(model, x)
                data = hidden[:, :noE] - K
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
