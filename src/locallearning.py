import math
import torch
from torch import nn
from torch import Tensor

class LocalLearningModel(nn.Module):
    class pSet():
        def __init__(self):
            self.in_size = 28**2
            self.hidden_size = 2000
            self.p = 2

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

    def __bracket(self, x:Tensor, y:Tensor) -> Tensor:
        eta = torch.pow(self.W, self.params.p - 2.0)
        eta = torch.abs(eta)
        res = torch.mul(x, eta)
        res = torch.matmul(res, y)
        return res

    def forward(self, x):
        x = self.flatten(x)
        hidden = self.__bracket(self.W, x)
        return hidden

if __name__ == "__main__":
    model_ps = LocalLearningModel.pSet()
    model = LocalLearningModel(model_ps)
    x = torch.rand((28, 28))
    print(model(x))
