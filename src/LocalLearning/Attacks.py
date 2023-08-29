from abc import ABC, abstractmethod

import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from .LocalLearning import HiddenLayerModel

# abstract baseclass interface for gradient based adversarial attacks
class AdversarialAttack(ABC):

    def __init__(self, model: HiddenLayerModel, mode="U", params=None):
        '''
        ARGS:   model: nn.Module - torch model
                mode: "U"/"T" - untargeted/targeted attack
                params: parameter dummy in case downstream implementation 
                        uses additional parameters (e.g. PGD)
        '''
        super(AdversarialAttack, self).__init__()
        self.model = model
        if mode == "U":
            self.grad_sign = +1.0   # gradient ascend from correct labels
        else:  
            self.grad_sign = -1.0   # gradient descent to wrong target label
        
        if params is not None:
            self.params = params
    
    @abstractmethod
    def create_examples(self, eps: float, data: Tensor, targets: Tensor, loss_fn: callable) -> Tensor:
        pass

    def __call__(self, eps: float, data: Tensor, targets: Tensor, loss_fn: callable) -> Tensor:
        adversarial_examples = self.create_examples(eps, data, targets, loss_fn)
        return adversarial_examples


class FGSM(AdversarialAttack):
    
    def __init__(self, model: nn.Module):
        super(FGSM, self).__init__(model)

    def create_examples(self, eps: float, data: Tensor, targets: Tensor, loss_fn: callable) -> Tensor:
        data.requires_grad = True
        pred, _ = self.model(data)
        loss = loss_fn(pred, targets)

        # calculate data gradient
        self.model.zero_grad()
        loss.backward()
        grad = data.grad.data

        # FGSM step
        perturbed_imgs = data + self.grad_sign*eps*grad.sign()
        perturbed_imgs = torch.clamp(perturbed_imgs, 0.0, 1.0)
        return perturbed_imgs


class PGD(AdversarialAttack):

    params = {
        "num_it": 10, # number of iterations,
        "dl": 1.0, # step size
    }
    
    def __init__(self, params: dict):
        super(PGD, self).__init__(params)

    def update(self, imgs, epsilon, grad):
        pass
