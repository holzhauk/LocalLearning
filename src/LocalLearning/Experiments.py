from abc import ABC, abstractmethod
from collections import OrderedDict
import pickle as pkl
from pathlib import Path

import torch
import numpy as np

from LocalLearning import HiddenLayerModel, ModelFactory
from LocalLearning.Attacks import AdversarialAttack, FGSM, PGD, WhiteGaussianPerturbation, AttackTest
from LocalLearning.Data import DeviceDataLoader, BaselineAccurateTestData

class PerturbationExperiment(ABC):
    
    def __init__(self, loss_fn: callable, batch_size: int = 1000):
        self.results = OrderedDict()
        self.loss_fn = loss_fn
        self.theFactory = ModelFactory()
        self.BATCH_SIZE = batch_size
        
    @abstractmethod
    def get_attack(self, model: HiddenLayerModel) -> AdversarialAttack:
        pass
    
    def run(self, directory: Path, model_fnlist: list, dataSet: torch.utils.data.Dataset, eps: np.array, device: torch.device, norm_p=2.0):
        for fn in model_fnlist:
            # load the model
            state_dict = torch.load(directory / fn)
            model = self.theFactory.build_from_state(state_dict)
            model.to(device)
            
            # generate data subset for baseline testing
            aTestData = BaselineAccurateTestData(model, dataSet)
            aTestLoader = DeviceDataLoader(
                aTestData,
                device=device,
                batch_size=self.BATCH_SIZE,
                num_workers=4,
                shuffle=False,
            )
            attack = self.get_attack(model)
            rpTest = AttackTest(attack, model, self.loss_fn)
            acc_rp, crit_eps_rp, crit_norm_rp = rpTest.run(aTestLoader, eps, norm_p=norm_p)
            
            self.results[str(fn)] = {}
            self.results[str(fn)]["eps"] = eps
            self.results[str(fn)]["no_imgs"] = len(aTestData)
            self.results[str(fn)]["acc"] = acc_rp
            self.results[str(fn)]["crit_eps"] = crit_eps_rp
            self.results[str(fn)]["crit_norm"] = crit_norm_rp
            
            del aTestData
            del aTestLoader
            del model
            del attack
            del rpTest
            
    def save(self, path: Path):
        with open(path, "ab") as file:
            pkl.dump(self.results, file)
            file.close()
        
    def load(self, path: Path):
        with open(path, "rb") as file:
            self.results = pkl.load(file)
            file.close()
            
    def get_result(self, key: str) -> dict:
        return self.results[key]
    
    def __iter__(self):
        self.keys = self.results.keys()
        self.keys_iter = iter(self.keys)
        return self
    
    def __next__(self):
        key = next(self.keys_iter)
        res = self.get_result(key)
        return (key, res)
    
class RandomPerturbationExperiment(PerturbationExperiment):
    def get_attack(self, model: HiddenLayerModel) -> AdversarialAttack:
        return WhiteGaussianPerturbation(model)
    
class FGSMExperiment(PerturbationExperiment):
    def get_attack(self, model: HiddenLayerModel) -> AdversarialAttack:
        return FGSM(model)
    
class PGDExperiment(PerturbationExperiment):
    
    pgd_params = {
        "num_it": 20,
        "dl": 1e-2,
        "dl_deps": 1e-1,
        "norm_p": 'inf',
    }
    
    def __init__(self, loss_fn: callable, pgd_params: dict=pgd_params):
        super().__init__(loss_fn)
        self.pgd_params = pgd_params.copy()
    
    def get_attack(self, model: HiddenLayerModel) -> AdversarialAttack:
        return PGD(model, self.pgd_params)
