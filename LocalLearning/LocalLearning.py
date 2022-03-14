import math
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
from torch import Tensor
from tqdm import tqdm


class LocalLearningModel(nn.Module):

    pSet = {
        "in_size": 28**2,
        "hidden_size": 2000,
        "n": 4.5,
        "p": 3,
        "tau_l": 10.0,
        "k": 7,
        "Delta": 0.4,
        "R": 1.0,
    }

    def __init__(self, params: dict):
        super(LocalLearningModel, self).__init__()

        self.pSet["in_size"] = params["in_size"]
        self.pSet["hidden_size"] = params["hidden_size"]
        self.pSet["n"] = params["n"]
        self.pSet["p"] = params["p"]
        self.pSet["tau_l"] = params["tau_l"]
        self.pSet["k"] = params["k"]
        self.pSet["Delta"] = params["Delta"]
        self.pSet["R"] = params["R"]

        with torch.no_grad():
            self.flatten = nn.Flatten()
            #  initialize weights
            self.W = nn.Parameter(
                torch.zeros((self.pSet["in_size"], self.pSet["hidden_size"])),
                requires_grad=False
            )
            # self.W = nn.Parameter(self.W) # W is a model parameter
            std = 1.0 / math.sqrt(self.pSet["in_size"] + self.pSet["hidden_size"])
            self.W.normal_(mean=0.0, std=std)
            
            self.relu = nn.ReLU()

    def __metric_tensor(self):
        eta = torch.abs(self.W)
        return torch.pow(eta, self.pSet["p"] - 2.0)

    def __bracket(self, v: Tensor, M: Tensor) -> Tensor:
        res = torch.mul(M, self.__metric_tensor())
        return torch.matmul(v, res)

    def __matrix_bracket(self, M_1: Tensor, M_2: Tensor) -> Tensor:
        res = torch.mul(M_1, self.__metric_tensor())
        res = torch.mul(M_2, res)
        return torch.sum(res, dim=0)

    def __g(self, q: Tensor) -> Tensor:
        g_q = torch.zeros(q.size(), device=self.W.device)
        sorted_idxs = q.argsort(descending=True)
        g_q[..., sorted_idxs[..., 0]] = 1.0
        g_q[..., sorted_idxs[..., 1 : self.pSet["k"] + 1]] = -self.pSet["Delta"]
        return g_q

    def __weight_increment(self, v: Tensor) -> Tensor:
        h = self.__bracket(v, self.W)
        Q = torch.pow(
            self.__matrix_bracket(self.W, self.W),
            (self.pSet["p"] - 1.0) / self.pSet["p"],
        )
        Q = torch.div(h, Q)
        inc = (self.pSet["R"] ** self.pSet["p"]) * v[..., None] - torch.mul(
            h[:, None, ...], self.W
        )
        inc = torch.mul(self.__g(Q)[:, None, ...], inc)
        return inc

    def forward(self, x):
        x = self.flatten(x)
        hidden = torch.matmul(x, self.W)
        output = torch.pow(self.relu(hidden), self.pSet["n"])
        return output 

    def param_dict(self) -> dict:
        return self.pSet

    def eval(self) -> None:
        pass

    def train(self, x: Tensor) -> None:
        # mean training, treating each mini batch as a sample:
        # dW = self.__weight_increment(x) / self.params.tau_l
        # dW_mean = torch.sum(dW, dim=0) / dW.size(dim=0)
        # self.W += dW_mean

        # sequential training in mini batch time:
        x_flat = self.flatten(x)
        for v in x_flat:
            v = v[None, ...] # single element -> minibatch of size 1
            self.W += self.__weight_increment(v).sum(dim=0) / self.pSet["tau_l"]


class BioLearningModel(nn.Module):

    pSet = {}

    def __init__(self, ll_trained_state: dict):
        super(BioLearningModel, self).__init__()

        local_learning_params = ll_trained_state["model_parameters"]
        self.local_learning = LocalLearningModel(local_learning_params)
        self.local_learning.load_state_dict(ll_trained_state["model_state_dict"])
        
        self.pSet = local_learning_params
        self.pSet["beta"] = 1.0
        self.dense = nn.Linear(self.pSet["hidden_size"], 10, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            hidden = self.local_learning(x)
        return self.tanh(self.pSet["beta"]*self.dense(hidden)) 



def train_unsupervised(
    dataloader: DataLoader, model: LocalLearningModel, device: torch.device, no_epochs=5, learning_rate=None
) -> None:
    '''
    Unsupervised learning routine for a LocalLearningModel on a PyTorch 
    dataloader

    learning_rate=None - constant learning rate according to tau_l provided in model
    learning_rate=float - constant learning rate learning_rate
    learning_rate=function - learning according to the functional relation specified by learning_rate(epoch)
    '''

    if type(learning_rate).__name__ != 'function':
        
        if type(learning_rate).__name__ == 'NoneType':
            learning_rate = 1.0 / model.pSet["tau_l"]
        
        learning_rate  = lambda l: learning_rate
    
    with torch.no_grad():
        for epoch in range(1, no_epochs):
            model.pSet["tau_l"] = 1.0 / learning_rate(epoch)
            with tqdm(dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch: {epoch}")
                for x, label in tepoch:
                    model.train(x.to(device))



if __name__ == "__main__":
    model_ps = LocalLearningModel.pSet()
    model = LocalLearningModel(model_ps)

    training_data = datasets.MNIST(
        root="../data/MNIST", train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="../data/MNIST", train=False, download=True, transform=ToTensor()
    )

    dataloader_train = DataLoader(training_data, batch_size=64)
    train_unsupervised(dataloader_train, model)
