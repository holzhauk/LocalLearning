from abc import ABC, abstractmethod

import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm.autonotebook import tqdm

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
    
    def __init__(self, model: HiddenLayerModel, mode="U"):
        super(FGSM, self).__init__(model, mode=mode)

    def create_examples(self, eps: float, data: Tensor, targets: Tensor, loss_fn: callable) -> Tensor:
        self.model.eval()
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
        return perturbed_imgs.detach()


class PGD(AdversarialAttack):

    params = {
        "num_it": 10, # number of iterations,
        "dl": 1.0, # step size
        "norm_p": "inf",
    }
    
    def __init__(self, model: HiddenLayerModel, params: dict, mode="U"):
        super(PGD, self).__init__(model, mode=mode, params=params)
        if type(self.params["norm_p"]) is str: # norm_p = 'inf' but also any other string type
            self._grad_step = self._Linf_grad
            self._L_ball_projection = self._Linf_ball_projection
        else:
            self._grad_step = self._Lp_grad
            self._L_ball_projection = self._Lp_ball_projection

    def _img_wise_Lp_norm(self, x: Tensor) -> Tensor:
        return torch.norm(torch.flatten(x, start_dim=1), p=self.params["norm_p"], dim=-1)
    
    def _img_wise_normalize(self, x: Tensor, factor: Tensor) -> Tensor:
        x = x.T
        x /= factor[..., :]
        return x.T

    def _L_ball_projection(self, data: Tensor, adv: Tensor, eps: Tensor) -> Tensor:
        pass

    def _grad_step(self, grad: Tensor) -> Tensor:
        pass

    def _Linf_grad(self, grad: Tensor) -> Tensor:
        return self.params["dl"]*grad.sign()
    
    def _Lp_grad(self, grad: Tensor) -> Tensor:
        # normalize each images gradient according to the Lp-norm
        norm_grad = self._img_wise_Lp_norm(grad)
        return self.params["dl"]*self._img_wise_normalize(grad, norm_grad)
    
    def _Linf_ball_projection(self, data: Tensor, adv: Tensor, eps: Tensor) -> Tensor:
        linf_radii_right = data.T + eps[..., :]
        linf_radii_left = data.T - eps[..., :]
        return torch.max(torch.min(adv, linf_radii_right.T), linf_radii_left.T)

    def _Lp_ball_projection(self, data: Tensor, adv: Tensor, eps: Tensor) -> Tensor:
        delta = adv - data
        dists = self._img_wise_Lp_norm(delta)
        # check for clipping event
        mask = (dists <= eps)
        # normalize
        scaling = dists
        scaling[mask] = eps # clipping -> projection onto Lp ball
        delta = self._img_wise_normalize(delta, scaling).T
        delta /= eps[..., :]
        return data + delta.T

    def create_examples(self, eps: float, data: Tensor, targets: Tensor, loss_fn: callable) -> Tensor:
        if eps < self.params["dl"]:
            dl = self.params["dl"]
            raise ValueError(f"eps bound can not be smaller than gradient update length dl: eps = {eps}.4f < {dl}.4f = dl")
        self.model.eval()
        adv = data.clone().detach().requires_grad_(True)
        eps = torch.ones((adv.size(dim=0),), device=data.device)*eps

        for _ in range(self.params["num_it"]):
            _adv = adv.clone().detach().requires_grad_(True)
            pred, _ = self.model(_adv)
            loss = loss_fn(pred, targets)

            # calculate data gradient
            self.model.zero_grad()
            loss.backward()
            grad = _adv.grad.data

            # gradient step
            adv = adv + self.grad_sign*self._grad_step(grad)
            # projection step
            #adv = self._L_ball_projection(data.clone().detach(), adv, eps)
        
        adv = torch.clamp(adv, 0.0, 1.0)
        return adv.detach()
            


class AttackTest():


    def __init__(
            self, 
            attack: AdversarialAttack, # attack method
            model: HiddenLayerModel,
            loss_fn: callable, # loss function
            ):
        self.model = model
        self.device = self._get_device()
        self.attack = attack
        self.loss_fn = loss_fn


    def _get_device(self) -> torch.device:
        dummy_d = next(iter(self.model.parameters()))
        for p in self.model.parameters():
            if p.device != dummy_d.device:
                raise RuntimeError(f"found that parameters on {type(self.model)} are on different devices")
        return dummy_d.device    

    def run(
            self, 
            loader: DataLoader, # data to test the model on
            eps: list, # list of perturbation sizes
            norm_p = 2.0, # image space distance norm
            dtype = torch.float32,
            ) -> tuple:
        
        # this implementation supports only sequential sampling
        if type(loader.sampler) is not torch.utils.data.sampler.SequentialSampler:
            raise TypeError("DataLoader loader does not use SequentialSampler for sampling from the data set. Please change sampler (e.g. by setting shuffle=True in DataLoader constructor).")


        no_images = len(loader.dataset)
        crit_eps = torch.zeros((no_images,), dtype=dtype, device=self.device)
        crit_norm = torch.zeros((no_images,), dtype=dtype, device=self.device)
        was_correct = torch.ones((no_images, ), dtype=torch.bool, device=self.device)
        accuracy = []

        with tqdm(eps, unit="perturbation") as teps:
            for eps in teps:
                teps.set_description(f"Testing dataset [epsilon={eps}.4f")

                freq = 0 # frequency of correct predictions

                for batch_nr, (data, targets)  in enumerate(loader):
                    
                    data.to(self.device)
                    targets.to(self.device)

                    # generate adversarial examples
                    self.model.eval()
                    adversarial_examples = self.attack.create_examples(
                        eps, 
                        data,
                        targets,
                        self.loss_fn,
                        )
                    
                    # test the model on adversarial examples
                    self.model.pred()
                    preds, hidden_reprs  = self.model(adversarial_examples)
                    correct = torch.isclose(
                        torch.abs(preds - targets),
                        torch.zeros(len(preds), dtype=torch.long, device=self.device), atol=1e-8
                        )
                    
                    # calculate and manage respective test metrics

                    # define batch window in the global index coordinates
                    bw_min = batch_nr*loader.batch_size
                    bw_max = (batch_nr + 1)*loader.batch_size

                    crit_event_mask = torch.logical_and(
                        was_correct[bw_min:bw_max], 
                        torch.logical_not(correct)
                        )
                    
                    was_correct[bw_min:bw_max][crit_event_mask] = False
                    crit_eps[bw_min:bw_max][crit_event_mask] = eps
                    crit_norm[bw_min:bw_max][crit_event_mask] = torch.norm(
                        torch.flatten(data - adversarial_examples, start_dim=1),
                        p=norm_p,
                        dim=-1,
                        )[crit_event_mask].clone().detach()

                    freq += float(correct.sum())
                    # clean up
                    del(preds)
                    del(correct)
                    del(adversarial_examples)
                    del(crit_event_mask)
                    del(hidden_reprs)
                    torch.cuda.empty_cache()

                accuracy.append(freq / no_images)


        return accuracy, \
                crit_eps.detach().cpu().clone().numpy(), \
                crit_norm.detach().cpu().clone().numpy()