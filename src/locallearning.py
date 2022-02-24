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
    class pSet():
        def __init__(self):
            self.in_size = 28**2 # default is MNIST
            self.hidden_size = 2000
            self.p = 2
            self.tau_l = 0.1 # local learning time scale
            self.k = 2
            self.Delta = 0.1 # inhibition rate
            self.R = 5.0

    def __init__(self, params: pSet):
        super(LocalLearningModel, self).__init__()
        self.params = params
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        #  initialize weights
        self.W = torch.zeros((self.params.hidden_size, \
                    self.params.in_size))
        #self.W = nn.Parameter(self.W) # W is a model parameter
        std = 1.0 / math.sqrt(self.params.in_size + \
                self.params.hidden_size)
        self.W.normal_(mean=0.0, std=std)

    def __metric_tensor(self):
        eta = torch.abs(self.W)
        return torch.pow(eta, self.params.p - 2.0)

    def __bracket(self, M: Tensor, v: Tensor) -> Tensor:
        res = torch.mul(M, self.__metric_tensor())
        return torch.matmul(res, v)

    def __matrix_bracket(self, M_1: Tensor, M_2: Tensor) -> Tensor:
        res = torch.mul(M_1, self.__metric_tensor())
        res = torch.mul(M_2, res)
        return torch.matmul(res, torch.ones(M_1.size(dim=1)))

    def __g(self, q: Tensor) -> Tensor:
        g_q = torch.zeros(q.size())
        sorted_idxs = q.argsort(descending=True)
        g_q[sorted_idxs[0]] = 1.0
        g_q[sorted_idxs[1:self.params.k + 1]] = -self.params.Delta
        return g_q

    def __weight_increment(self, x: Tensor) -> Tensor:
        h = self.forward(x)
        x = self.flatten(x)
        Q = torch.div(h, self.__matrix_bracket(self.W, self.W))
        inc = (self.params.R**self.params.p)*x[None, ...] - \
                torch.mul(h[..., None], self.W)
        inc = torch.mul(self.__g(Q)[..., None], inc)
        return inc

    def forward(self, x):
        x = self.flatten(x)
        print(x.size())
        hidden = self.__bracket(self.W, x)
        return hidden

    def update_weights(self, x: Tensor) -> None:
        self.W = self.W + self.__weight_increment(x) / self.params.tau_l

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def LocalLearn(dataloader, model: LocalLearningModel) -> None:
    with torch.no_grad():
        dSet_size = len(dataloader.dataset)
        for batch, (x, label) in enumerate(dataloader):
            #print(x.size())
            #print(label)
            print(model(x).size())
            #model.update_weights(x)            

if __name__ == "__main__":
    model_ps = LocalLearningModel.pSet()
    model = LocalLearningModel(model_ps)

    training_data = datasets.MNIST(
        root="../data/MNIST",
        train=True,
        download=True,
        transform=ToTensor()
            )
    
    test_data = datasets.MNIST(
        root="../data/MNIST",
        train=False,
        download=True,
        transform=ToTensor()
            )
    
    dataloader_train = DataLoader(training_data, batch_size=64)
    nn = NeuralNetwork()

    LocalLearn(dataloader_train, nn)
