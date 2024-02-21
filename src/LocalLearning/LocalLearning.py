'''
Copyright 2018 Dmitry Krotov

PyTorch Implementation and further modifications:

Copyright 2024 Konstantin Holzhausen and University of Oslo

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    PyTorch implementation of Biological Learning described in [3].

    [3] Dmitry Krotov and John J. Hopfield, "Unsupervised learning by competing hidden units.", 2019.
        PNAS, Vol. 116, No. 16 (https://doi.org/10.1073/pnas.1820458116)

    repository: https://github.com/DimaKrotov/Biological_Learning
    commit: 45478bb8143cc6aa3984c040dfd1bc8bc44e4e29

    A copy of the Apache License is found in the LICENSE.Apache2 file in this directory
'''

from abc import ABC, abstractmethod

import math
from pathlib import Path
from tqdm.autonotebook import tqdm

from collections import OrderedDict
import copy

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import Tensor
from torch.optim import Adam


class KHL3(nn.Module):
    """
    Krotov and Hopfield's (KH) Local Learning Layer (L3)
    as first implemented by Konstantin Holzhausen
    CAREFUL - definition of g does not matches any of the ones mentioned in the paper
    """

    pSet = {
        "in_size": 28 ** 2,
        "hidden_size": 2000,
        "n": 4.5,
        "p": 3,
        "tau_l": 1 / 0.04,
        "k": 7,
        "Delta": 0.4,
        "R": 1.0,
    }

    def __init__(self, init_dict: dict, sigma=None):
        '''
        ARGS:
            init_dict (dict):   either
                                (I) - init_dict = state_dict: initialising KHL3 type model
                                        - from serialized form, loading parameter values
                                (II) - init_dict = pSet: initialising KHL3 based on description
                                        - random values
        '''
        super().__init__()

        self.flatten = nn.Flatten()
        self.flatten.requires_grad_(False)
        
        # differentiate between both forms of initialization (I, II) based on 
        # structure of the dict
        try: # assume init_dict is torch state dict
            if init_dict['type_name'] != type(self).__name__:
                raise IOError(f"state_dict does not correspond to {type(self).__name__} model")
            else:
                self._write_pSet(init_dict["pSet"])
                self.W = nn.Parameter(
                    torch.empty((self.pSet["in_size"], self.pSet["hidden_size"])),
                    requires_grad=False,
                )
                self.load_state_dict(init_dict)
        except KeyError: # otherwise init_dict is pSet
            parameter_set = init_dict
            self._write_pSet(parameter_set)

            #  initialize weights
            self.W = nn.Parameter(
                torch.zeros((self.pSet["in_size"], self.pSet["hidden_size"])),
                requires_grad=False,
            )
            # self.W = nn.Parameter(self.W) # W is a model parameter
            if type(sigma) == type(None):
                # if sigma is not explicitely specified, use Glorot
                # initialisation scheme
                sigma = 1.0 / math.sqrt(self.pSet["in_size"] + self.pSet["hidden_size"])

            self.W.normal_(mean=0.0, std=sigma)

    def __metric_tensor(self):
        eta = torch.abs(self.W)
        return torch.pow(eta, self.pSet["p"] - 2.0)

    def _bracket(self, v: Tensor, M: Tensor) -> Tensor:
        res = torch.mul(M, self.__metric_tensor())
        return torch.matmul(v, res)

    def __matrix_bracket(self, M_1: Tensor, M_2: Tensor) -> Tensor:
        res = torch.mul(M_1, self.__metric_tensor())
        res = torch.mul(M_2, res)
        return torch.sum(res, dim=0)

    def __g(self, q: Tensor) -> Tensor:
        g_q = torch.zeros(q.size(), device=self.W.device)
        _, sorted_idxs = q.topk(self.pSet["k"], dim=-1)
        batch_size = g_q.size(dim=0)
        g_q[range(batch_size), sorted_idxs[:, 0]] = 1.0
        g_q[range(batch_size), sorted_idxs[:, 1:]] = -self.pSet["Delta"]
        return g_q

    def __weight_increment(self, v: Tensor) -> Tensor:
        h = self._bracket(v, self.W)
        Q = torch.pow(
            self.__matrix_bracket(self.W, self.W),
            (self.pSet["p"] - 1.0) / self.pSet["p"],
        )
        Q = torch.div(h, Q)
        inc = (self.pSet["R"] ** self.pSet["p"]) * v[..., None] - torch.mul(
            h[:, None, ...], self.W
        )
        return torch.mul(self.__g(Q)[:, None, ...], inc).sum(dim=0)

    def forward(self, x):
        x_flat = self.flatten(x)
        return self._bracket(x_flat, self.W)

    def param_dict(self) -> dict:
        return self.pSet
    
    def _write_pSet(self, params: dict) -> None:
        self.pSet["in_size"] = params["in_size"]
        self.pSet["hidden_size"] = params["hidden_size"]
        self.pSet["n"] = params["n"]  # not used in this class, but belongs to the model
        self.pSet["p"] = params["p"]
        self.pSet["tau_l"] = params["tau_l"]
        self.pSet["k"] = params["k"]
        self.pSet["Delta"] = params["Delta"]
        self.pSet["R"] = params["R"]
    
    def state_dict(self, *args, **kwargs) -> dict:
        state_dict = super(KHL3, self).state_dict(*args, **kwargs)
        state_dict["pSet"] = self.pSet
        state_dict["type_name"] = type(self).__name__
        return state_dict
    
    def load_state_dict(self, state_dict: dict, *args, **kwargs) -> None:
        if state_dict["type_name"] != type(self).__name__:
            raise IOError(f"state_dict does not correspond to {type(self).__name__} model")
        dummy = state_dict.copy()
        del dummy["type_name"]

        self._write_pSet(dummy["pSet"])
        del dummy["pSet"]

        super(KHL3, self).load_state_dict(dummy, *args, **kwargs)

    def eval(self) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        pass

    def train_step(self, x: Tensor) -> None:
        # mean training, treating each mini batch as a sample:
        # dW = self.__weight_increment(x) / self.params.tau_l
        # dW_mean = torch.sum(dW, dim=0) / dW.size(dim=0)
        # self.W += dW_mean

        # sequential training in mini batch time:
        x_flat = self.flatten(x)
        for v in x_flat:
            v = v[None, ...]  # single element -> minibatch of size 1
            self.W += self.__weight_increment(v) / self.pSet["tau_l"]


class FKHL3(KHL3):
    """
    Fast AI implementation (F) of KHL3
    """

    def __init__(self, params: dict, sigma=None):
        super(FKHL3, self).__init__(params, sigma)

    # redefining the relevant routines to make them fast
    # "fast" means that it allows for parallel mini-batch processing

    def __g(self, q: Tensor) -> Tensor:
        g_q = torch.zeros(q.size(), device=self.W.device)
        _, sorted_idxs = q.topk(self.pSet["k"], dim=-1)
        batch_size = g_q.size(dim=0)
        g_q[range(batch_size), sorted_idxs[:, 0]] = 1.0
        g_q[range(batch_size), sorted_idxs[:, -1]] = -self.pSet["Delta"]
        return g_q

    def __weight_increment(self, v: Tensor) -> Tensor:
        h = self._bracket(v, self.W)
        g_mu = self.__g(h)
        inc = self.pSet["R"] ** self.pSet["p"] * (v.T @ g_mu)
        return inc - (g_mu * h).sum(dim=0)[None, ...] * self.W

    def train_step(self, x: Tensor, prec=1e-9) -> None:
        # implementation of the fast unsupervised
        # training algorithm
        # it is fast because it does not require sequential training over
        # minibatch

        x_flat = self.flatten(x)
        dW = self.__weight_increment(x_flat)
        nc = max(dW.abs().max(), prec)
        self.W += dW / (nc * self.pSet["tau_l"])


class HiddenLayerModel(nn.Module, ABC):
    
    pSet = {
        "hidden_size": 2000,
    }
    
    def __init__(self):
        super().__init__()
        setattr(self, 'forward', self._forward)
        self.device = torch.device('cpu')

    @abstractmethod
    def hidden(self, x: torch.Tensor):
        pass

    @abstractmethod
    def _forward(self, x: torch.Tensor):
        pass

    def pred(self):
        def preds(x: torch.Tensor) -> torch.Tensor:
            logits, hidden = self._forward(x)
            return (torch.argmax(logits, dim=-1), hidden)
        setattr(self, 'forward', preds)

    def eval(self):
        super().eval()
        setattr(self, 'forward', self._forward)

    def train(self, val=True):
        super().train(val)
        setattr(self, 'forward', self._forward)

    def to(self, dev: torch.device):
        super().to(dev)
        p0 = next(iter(self.parameters()))
        self.device = p0.device
        


class KHModel(HiddenLayerModel):
    """
    Model similar to that propsed by Krotov and Hopfield (KH)
    Architectural structure:
         FKHL3 (Fast Local Learning Layer)
           |
         PReLu (Polynomial ReLu)
           |
         Dense (Linear)
    """

    pSet = {}

    def __init__(self, *args, **kwargs): #ll_trained_state: dict):
        super().__init__()

        if len(args) < 1:
            raise IOError("'KHModel' constructor does not accept less than 1 argument")
        
        self.relu_h = nn.ReLU()
        self.relu_h.requires_grad_(False)

        self.softMax = nn.Softmax(dim=-1)

        if type(args[0]).__name__ == OrderedDict.__name__:
            state_dict = args[0]
            if state_dict["type_name"] != type(self).__name__:
                raise IOError(f"state_dict does not correspond to {type(self).__name__} model")
            self.local_learning = FKHL3(state_dict["pSet"]["FKHL3_pSet"])

            self.dense = nn.Linear(state_dict["pSet"]["FKHL3_pSet"]["hidden_size"], state_dict["pSet"]["no_classes"])

            self.load_state_dict(state_dict)

        elif issubclass(type(args[0]), FKHL3):
            self.local_learning = args[0]
            self.pSet = copy.deepcopy(self.local_learning.pSet)

            # second argument must be keyword and has to be number of classification units
            if ("no_classes" not in kwargs.keys()) or  ((type(kwargs["no_classes"])) is not int):
                raise IOError(f"First constructor element is of type {type(args[0]).__name__}" + 
                              f", but second is not no_classes='int' but {kwargs} of type {type(args[1]).__name__}") 
            self.pSet["no_classes"] = kwargs['no_classes']

            self.dense = nn.Linear(self.pSet["hidden_size"], self.pSet["no_classes"])

        else:
            raise TypeError("'KHModel' constructor does not accept arguments of this type")

        self.local_learning.requires_grad_(False)
        self.dense.requires_grad_(True)
        

    def state_dict(self, *args, **kwargs) -> dict:
        state_dict = super().state_dict(*args, **kwargs)
        dummy = self.pSet.copy()
        state_dict["pSet"] = {}
        state_dict["pSet"]["no_classes"] = dummy["no_classes"]
        del dummy["no_classes"]
        state_dict["pSet"]["FKHL3_pSet"] = dummy.copy()
        state_dict["type_name"] = type(self).__name__
        return state_dict
    
    def load_state_dict(self, state_dict: dict, *args, **kwargs) -> None:
        if state_dict["type_name"] != type(self).__name__:
            raise IOError(f"state_dict does not correspond to {type(self).__name__} model")
        dummy = state_dict.copy()
        del dummy["type_name"]

        pSet =dummy["pSet"]["FKHL3_pSet"].copy()
        pSet["no_classes"] = dummy["pSet"]["no_classes"]
        self.pSet = pSet
        del dummy["pSet"]

        super().load_state_dict(dummy, *args, **kwargs)


    def hidden(self, x: torch.Tensor) -> Tensor:
        return self.local_learning(x)

    def _forward(self, x: Tensor) -> Tensor:
        hidden = self.hidden(x)
        latent_activation = torch.pow(self.relu_h(hidden), self.pSet["n"])
        return (self.dense(latent_activation), hidden)


class SHLP(HiddenLayerModel):
    # single hidden layer perceptron model
    
    # default parameters
    pSet = {
        "in_size": 32*32*3,
        "hidden_size": 2000,
        "n": 4.5,
        "no_classes": 10,
        "batch_norm": False,
    }

    def __init__(self, in_dict: dict=None, sigma: float=None, batch_norm=False, dtype: torch.dtype=torch.float32, **kwargs):
        super(SHLP, self).__init__()
        
        load_state = False

        if "type_name" in in_dict.keys():
            # in_dict is state dict
            state_dict = in_dict.copy()
            self.pSet = state_dict["pSet"].copy()
            load_state = True
        else:
            # in_dict is parameter dictionary
            params = in_dict.copy()
            if type(params) != type(None):
                self.pSet["in_size"] = params["in_size"]
                self.pSet["hidden_size"] = params["hidden_size"]
                self.pSet["n"] = params["n"]
                self.pSet["no_classes"] = params["no_classes"]
                self.pSet["batch_norm"] = batch_norm
            
        self.dtype = dtype
        self.flatten = nn.Flatten()
        
        # define linear mapping between input and hidden layer
        # creating the representations
        # same fashion as in KHL3 for optimal control
        self.W = torch.zeros((self.pSet["in_size"], self.pSet["hidden_size"]), dtype=self.dtype)

        # if sigma not explicitely specified, use Keiming He initialisation scheme
        # from: He et al. (2015) "Delving Deep into Rectifiers: 
        # Surpassing Human-Level Performance on ImageNet Classification"
        if type(sigma) == type(None):
            sigma = math.sqrt(2 / self.pSet["in_size"])

        self.W.normal_(mean=0.0, std=sigma)
        self.W = nn.Parameter(self.W)
        
        self.ReLU = nn.ReLU()
        # define second mapping
        modules = []
        if self.pSet["batch_norm"]:
            modules.append(nn.BatchNorm1d(self.pSet["hidden_size"]))
        modules.append(nn.Linear(self.pSet["hidden_size"], self.pSet["no_classes"]))
        self.dense = nn.Sequential(*modules)

        if load_state:
            self.load_state_dict(state_dict)
        
    def hidden(self, x: torch.Tensor):
        x_flat = self.flatten(x)
        return x_flat @ self.W
        
    def _forward(self, x: torch.Tensor):
        hidden = self.hidden(x)
        latent_activation = torch.pow(self.ReLU(hidden), self.pSet["n"])
        return self.dense(latent_activation), hidden
    
    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict["pSet"] = self.pSet
        state_dict["type_name"] = type(self).__name__
        return state_dict
    
    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict["type_name"] != type(self).__name__:
            raise IOError(f"state_dict does not correspond to {type(self).__name__} model")
        dummy = state_dict.copy()
        del dummy["type_name"]

        self.pSet = dummy["pSet"].copy()
        del dummy["pSet"]

        super().load_state_dict(dummy)
        
class SHLP_tanh(SHLP):
    # single hidden layer perceptron model
    
    def __init__(self, in_dict: dict=None, sigma: float=None, batch_norm=False, dtype: torch.dtype=torch.float32, **kwargs):
        super(SHLP_tanh, self).__init__(in_dict=in_dict, sigma=sigma, batch_norm=batch_norm, dtype=dtype, **kwargs)
        self.ReLU = nn.Tanh() 


class IdentityModel(nn.Module):

    pSet = {
        "hidden_size": 2000,
    }
    
    def __init__(self, params: dict, **kwargs):
        super(IdentityModel, self).__init__()
        self.pSet["hidden_size"] = params["in_size"]
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        return self.flatten(x)
    

class ModelFactory():
    def build_from_state(self, state_dict: dict) -> HiddenLayerModel:
        '''
        Creates and returns a HiddenLayerModel based on the loaded state_dict.
        Chooses the subtype based on the descriptive field "type_name" in state_dict
        '''
        try:
            type_name = state_dict["type_name"]
        except KeyError:
            print("state_dict not supported. " + 
                  "Misses field 'type_name' identifying the subtype of HiddenLayerModel to choose")

        if state_dict["type_name"] == KHModel.__name__:
            return KHModel(state_dict)
        elif state_dict["type_name"] == SHLP.__name__:
            return SHLP(state_dict)
        elif state_dict["type_name"] == SHLP_tanh.__name__:
            return SHLP_tanh(state_dict)
        elif state_dict["type_name"] == FKHL3.__name__:
            return FKHL3(state_dict)
        else:
            raise ValueError(f"Model type: {type_name} not supported")


def train_unsupervised(
    dataloader: DataLoader,
    model: KHL3,
    device: torch.device,
    filepath: Path,
    no_epochs=5,
    checkpt_period=1,
    learning_rate=None,
) -> None:
    """
    Unsupervised learning routine for a LocalLearningModel on a PyTorch 
    dataloader

    learning_rate=None - constant learning rate according to tau_l provided in model
    learning_rate=float - constant learning rate learning_rate
    learning_rate=function - learning according to the functional relation specified by learning_rate(epoch)
    """

    if type(learning_rate).__name__ != "function":

        if type(learning_rate).__name__ == "NoneType":
            learning_rate = 1.0 / model.pSet["tau_l"]

        lr = lambda l: learning_rate

    else:
        lr = learning_rate

    with torch.no_grad():
        with tqdm(range(1, no_epochs + 1), unit="epoch") as tepoch:
            #tepoch.set_description(f"Epoch: {epoch}")
            tepoch.set_description(f"Training time [epochs]")

            for epoch in tepoch:
                # catch lr(epoch) = 0 to avoid division by 0
                if lr(epoch) != 0.0:
                    # if learning rate == 0 -> no learning
                    model.pSet["tau_l"] = 1.0 / lr(epoch)
                    for batch_num, (features, labels) in enumerate(dataloader):
                        model.train_step(features.to(device))

                if epoch % checkpt_period == 0:
                    torch.save(
                        {
                            "model_parameters": model.param_dict(),
                            "model_state_dict": model.state_dict(),
                            "device_type": device.type,
                        },
                        filepath.parent
                        / Path(
                            str(filepath.stem) + "_" + str(epoch) + str(filepath.suffix)
                        ),
                    )


def train_half_backprop(
    dataloader: DataLoader,
    model: KHModel,
    device: torch.device,
    filepath: Path,
    no_epochs=5,
    checkpt_period=1,
    learning_rate=None,
) -> None:
    if type(learning_rate).__name__ != "function":

        if type(learning_rate).__name__ == "NoneType":
            learning_rate = 1.0 / model.pSet["tau_l"]

        learning_rate = lambda l: learning_rate

    def loss_fn(x: Tensor, labels: Tensor) -> float:
        return torch.mean(torch.pow(x - labels, m))

    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(1, no_epochs):

        cummulative_loss = 0.0
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch: {epoch}")
            # for x, label in tepoch:


def weight_convergence_criterion(model: KHL3, conv_tol: float, tol: float) -> bool:
    '''
    Checks whether the weights in have converged or not according to the criterion specified.
    Unit is considered converged if the norm of the respective weights is closer at R than conv_tol.
    If the fraction of converged units is larger equal tol, the model is considered converged.

    ARGS:
        model (KHL3):   Krotov and Hopfield Local Learning Model
        conv_tol (float):   absolute tollerance in norm
        tol (float):    relative tollerance regarding the fraction of converged units

    VALS:
        return (bool):  True if criterion is met
    '''
    p = model.pSet["p"]
    R = model.pSet["R"]

    ds = torch.abs(torch.norm(model.W, p=p, dim=0) - R)
    fraction = (ds > conv_tol).sum() / model.pSet["hidden_size"]
    return (fraction < tol)

def weight_mean_criterion(model: KHL3) -> bool:
    '''
    Checks whether the weights are reasonable sized.
    In rare cases, the training dynamics can diverge, 
    which results in all negative weights close to -R for individual units. 
    Therefore, we check whether the mean weight of each unit is reasonable
    far away from -R. 
    '''
    return ((model.W.mean(dim=0) < -model.pSet["R"] / 2.0).sum().item() == 0)