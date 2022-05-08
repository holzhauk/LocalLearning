import math
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
from torch import Tensor
from torch.optim import Adam


class LpUnitCIFAR10(datasets.CIFAR10):
    def __init__(self, root, transform, train=True, p=2.0):
        super(LpUnitCIFAR10, self).__init__(
            root=root, transform=transform, train=train, download=True
        )
        self.p = p

    def __len__(self):
        return super(LpUnitCIFAR10, self).__len__()

    def __getitem__(self, index):
        feature, label = super(LpUnitCIFAR10, self).__getitem__(index)
        # normalize image to a Lp unit vector
        feature /= torch.norm(feature, p=self.p)
        return feature, label


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
        "tau_l": 10.0,
        "k": 7,
        "Delta": 0.4,
        "R": 1.0,
    }

    def __init__(self, params: dict):
        super(KHL3, self).__init__()

        self.pSet["in_size"] = params["in_size"]
        self.pSet["hidden_size"] = params["hidden_size"]
        self.pSet["n"] = params["n"]  # not used in this class, but belongs to the model
        self.pSet["p"] = params["p"]
        self.pSet["tau_l"] = params["tau_l"]
        self.pSet["k"] = params["k"]
        self.pSet["Delta"] = params["Delta"]
        self.pSet["R"] = params["R"]

        self.flatten = nn.Flatten()
        self.flatten.requires_grad_(False)
        #  initialize weights
        self.W = nn.Parameter(
            torch.zeros((self.pSet["in_size"], self.pSet["hidden_size"])),
            requires_grad=False,
        )
        # self.W = nn.Parameter(self.W) # W is a model parameter
        std = 1.0 / math.sqrt(self.pSet["in_size"] + self.pSet["hidden_size"])
        self.W.normal_(mean=0.0, std=std)

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

    def __init__(self, params: dict):
        super(FKHL3, self).__init__(params)

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

    def train_step_fast(self, x: Tensor, prec=1e-30) -> None:
        # implementation of the fast unsupervised
        # training algorithm
        # it is fast because it does not require sequential training over
        # minibatch

        x_flat = self.flatten(x)
        dW = self.__weight_increment(x_flat)
        nc = max(dW.abs().max(), prec)
        self.W += dW / (nc * self.pSet["tau_l"])


class BioLearningModel(nn.Module):

    pSet = {}

    def __init__(self, ll_trained_state: dict):
        super(BioLearningModel, self).__init__()

        self.pSet = ll_trained_state["model_parameters"]
        self.local_learning = KHL3(self.pSet)
        self.local_learning.load_state_dict(ll_trained_state["model_state_dict"])

        self.relu = nn.ReLU()
        self.relu.requires_grad_(False)

        self.dense = nn.Linear(self.pSet["hidden_size"], 10, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.local_learning(x)
        activation = self.relu(h)
        # activation = torch.pow(self.relu(h), self.pSet["n"])
        # return self.softmax(self.dense(activation))
        return activation


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
        for epoch in range(1, no_epochs):
            model.pSet["tau_l"] = 1.0 / lr(epoch)
            with tqdm(dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch: {epoch}")
                for x, label in tepoch:
                    model.train_step(x.to(device))

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
    model: BioLearningModel,
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
