import torch
from torch import Tensor
from LocalLearning import HiddenLayerModel
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from copy import deepcopy

import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2

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
    

class StatisticGardener():
    '''
    class implementing the pruning routine for the locally learned weights

    description:

        Decision to prune is based on the distribution of log-variances
        along the hidden dimension in the weights.
        We use model selection by likelihood ratio to detect the number of 
        gaussian components in the distribution. 
        If the distribution is unimodal, pruning is considered not necessary.
        In case the distribution is multi-modal, we prune the contributions that 
        account for the component with the highest variance.
    '''

    def __init__(self):
        self.heuristic_means = np.array([[-7.1], [-8.5], [-6.5], [-6.0]])
        self.no_components = 1 # number of Gaussian components
        self.log_normal_mixture = None

    def __log_likelihood(self, data: np.ndarray, c: int) -> np.ndarray:
        '''
        calculates the log-likelihood of a gaussian mixture model with c components
        
        ARGS:
            data (np.ndarray):  data array
            c (int):    number of components

        VALS:
            return (np.ndarray):    log-likelihood value
        '''
        if c == 1:
            # unimodal model
            mu, sigma = np.mean(data), np.std(data)
            return np.sum(norm.logpdf(data, mu, sigma)) # log-likelihood of currently selected model
        
        self.log_normal_mixture = GaussianMixture(
            n_components = c,
            means_init = self.heuristic_means[:c],
            max_iter = 2000,
            random_state=0,
            )
        self.log_normal_mixture.fit(data.reshape(-1, 1))
        return self.log_normal_mixture.score(data.reshape(-1, 1)) * len(data) # log likelihood of more complex model


    def prune(self, model: KHL3) -> tuple:
        '''
        pruning based on modality of the log-variance distribution

        ARGS:
            model (LL.KHL3):    local-learning model to be pruned

        VALS:
            return (tuple):   (
                                pruned model: LL.KHL3,
                                no_components: int, - number of Gussian components detected
                                cutoff, - cutoff value, None in case of unimodal distribution

                                )
        '''
        W = model.W.clone()
        log_data = torch.log(W.var(dim=0)).detach().cpu().numpy()
        # clip data to heuristic range
        log_data = log_data[log_data < -3.0]
        log_data = log_data[log_data > -9.0] 

        # unimodal model
        n_components = 1 # number of gaussian components
        log_l = self.__log_likelihood(log_data, n_components)

        # Compare multi-modal model to uni-modal model
        # select model up to four components
        max_n = len(self.heuristic_means)
        searching = True
        while searching and (n_components < max_n):
            # define and infer log normal mixture model (GaussianMixture on log data)
            n_components += 1
            log_l_c = self.__log_likelihood(log_data, n_components)

            # likeihood ratio statistics
            lambda_LR = -2 * (log_l - log_l_c)
            p_value = 1 - chi2.cdf(lambda_LR, 3) # each higher order gaussian mixture has three more parameters than the previous one

            # selection criterion
            if (p_value < 0.01): 
                # reject null hypothesis, new model is the best, continue searching
                if n_components <  max_n:
                    log_l = log_l_c
                    searching = True
            else:
                # accept null hypothesis, old model is better, stop searching model selected
                searching = False

        model_pruned = deepcopy(model)

        if n_components == 1:
            # no pruning necessary, distribution of weights is unimodal
            return (model_pruned, n_components, None)
        else:
            # pruning necessary
            means = self.log_normal_mixture.means_.reshape(1, -1)[0]
            idxs = np.argsort(means)
            mu_noise = means[idxs[-1]]
            mu_imgs = means[idxs[-2]]
            cutoff = mu_noise - np.abs(mu_noise - mu_imgs) / 2.0

            model_pruned.W[:, (W.var(dim=0) >= np.exp(cutoff))] = 0.0
            return (model_pruned, n_components, np.exp(cutoff))
