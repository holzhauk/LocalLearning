from abc import ABC, abstractmethod

import numpy as np

from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from tqdm.autonotebook import tqdm

from .LocalLearning import HiddenLayerModel

# define the training interface
class Trainer(ABC):

    class Logger():
        def __init__(self, model: HiddenLayerModel=None):
            self.model_type = type(model).__name__
            self.model_path = Path("")
            self.log = {}

        def __getitem__(self, key: str):
            return self.log[key]
        
        def __setitem__(self, key: str, val):
            self.log[key] = val

        def __delitem__(self, key: str):
            del self.log[key]

        def save(self, path: Path):
            json_object = json.dumps(vars(self))
            with open(path, "w") as file:
                file.write(json_object)

        def load(self, path: Path):
            with open(path, "r") as file:
                json_obj = json.load(file)
                self.model_type = json_obj["model_type"]
                self.model_path = json_obj["model_path"]
                self.log = json_obj["log"]

        def keys(self) -> list:
            return self.log.keys()


    def __init__(
            self,
            model: nn.Module,
            ):
        super(Trainer, self).__init__()
        self.model = model
        self.log = Trainer.Logger(self.model)

    @abstractmethod
    def run(
            self,
            dataTrain: DataLoader,
            dataTest: DataLoader=None,
            ) -> None:
        pass

    @abstractmethod
    def _epoch_preprocessing_train(
            self,
            ) -> None:
        pass

    @abstractmethod
    def _epoch_postprocessing_train(
            self,
            ) -> None:
        pass

    @abstractmethod
    def _epoch_preprocessing_eval(
            self,
            ) -> None:
        pass

    @abstractmethod
    def _epoch_postprocessing_eval(
            self,
            ) -> None:
        pass

    @abstractmethod
    def _batch_preprocessing(
            self,
            features: torch.Tensor,
            labels: torch.Tensor,
            ) -> torch.Tensor:
        pass

    @abstractmethod
    def _batch_loss(
            self,
            features: torch.Tensor,
            labels: torch.Tensor,
            outputs: torch.Tensor,
            hidden_repr: torch.Tensor,
            ) -> torch.Tensor:
        pass

    @abstractmethod
    def _batch_eval(
            self,
            features: torch.Tensor,
            labels: torch.Tensor,
            predictions: torch.Tensor,
            hidden_repr: torch.Tensor,
            ) -> float:
        pass

    def save(self, model_path: Path, log_path: Path):
        self.log.model_path = str(model_path)
        torch.save(self.model.state_dict(), model_path)
        self.log.save_log(log_path)


class CETrainer(Trainer):
    
    #sPs = {
    #    "lambda_JR": 1e-3, # Lagrange Multiplier for Jacobian Regularisation
    #    "n": 1, # Number of Projections 
    #}
    
    ce_loss_fn = nn.CrossEntropyLoss()

    def __init__(self,
                 model: HiddenLayerModel,
                 device: torch.device,
                 #spectral_ps: sPs=sPs,
                 learning_rate: float=1e-4,
                 dtype: torch.dtype=torch.float32,
                ):
        super(CETrainer, self).__init__(model)
        #self.no_epochs = no_epochs
        #self.sPs = spectral_ps
        self.device = device
        self.dtype = dtype

        if type(learning_rate) is float:
            self.learning_rate = lambda epoch: learning_rate
        elif type(learning_rate) is callable:
            self.learning_rate = learning_rate
        else:
            raise ValueError("learning_rate neither 'float' nor 'callable'")
        
        
        # History Logging Functionality
        self.log["epoch"] = []
        self.log["loss"] = []
        self.log["ce_loss"] = []
        self.log["eval_acc"] = []
        
        #self.JacReg = JacobianReg(self.sPs["n"])
    
    def run(
            self, 
            trainData: DataLoader,
            testData: DataLoader=None,
            no_epochs: int=5,
                ):
        # make dataloader accessible for decorators as well
        self.trainData = trainData
        self.testData = testData
        optimizer = Adam(self.model.parameters(), lr=1.0)
        scheduler = LambdaLR(optimizer, lr_lambda=[self.learning_rate])
    
        with tqdm(range(1, no_epochs + 1), unit="epoch") as tepoch:
            tepoch.set_description(f"Training time [epochs]")
            for epoch in tepoch:
                
                # epoch preprocessing
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

                self._epoch_preprocessing_train()
                self.model.to(self.device)
                self.model.train()
                cumm_loss = 0.0   

                for batch_nr, (features, labels) in enumerate(self.trainData):

                    # batch preprocessing
                    features, labels = self._batch_preprocessing(features, labels)
                    
                    # batch optimization
                    optimizer.zero_grad()
                    outputs, hidden_repr = self.model(features)
                    loss = self._batch_loss(features, labels, outputs, hidden_repr)
                    cumm_loss += float(loss)

                    loss.backward()
                    optimizer.step()
                    # gradient clipping
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, model.pSet["p"])
                    
                    # batch post-processing
                    self._batch_postprocessing(features, labels)
                 
                # epoch postprocessing
                self.log["epoch"].append(epoch)
                self.log["loss"].append(cumm_loss)
                self._epoch_postprocessing_train()
                
                if testData is not None:
                    self._epoch_preprocessing_eval()
                    self.model.pred()
                    eval_freq = 0.0 # evaluation frequency
                    for batch_no, (features, labels) in enumerate(self.testData):
                        # batch preprocessing
                        features, labels = self._batch_preprocessing(features, labels)
                        
                        # batch evaluation
                        predictions, hidden_repr = self.model(features)
                        eval_freq += self._batch_eval(features, labels, predictions, hidden_repr)
                    
                    self.log["eval_acc"].append(eval_freq / (len(testData)*testData.batch_size))
                    self._epoch_postprocessing_eval()

                scheduler.step()

    def _epoch_preprocessing_train(
            self,
            ):
        self.cumm_ce_loss = 0.0

    def _epoch_postprocessing_train(
            self,
            ):
        self.log["ce_loss"].append(self.cumm_ce_loss)

    def _epoch_preprocessing_eval(
            self,
            ) -> None:
        return None

    def _epoch_postprocessing_eval(
            self,
            ) -> None:
        return None

    def _batch_preprocessing(
            self,
            features: torch.Tensor,
            labels: torch.Tensor,
            ) -> tuple:

        features = features.to(self.device)
        labels = labels.to(self.device)
        return (features, labels)

    def _batch_postprocessing(
            self,
            features: torch.Tensor,
            labels: torch.Tensor
            ) -> None:
        
        features.to('cpu', non_blocking=True)
        labels.to('cpu', non_blocking=True)

    def _batch_loss(
            self,
            features: torch.Tensor,
            labels: torch.Tensor,
            outputs: torch.Tensor,
            hidden_repr: torch.Tensor,
            ) -> torch.Tensor:

        ce_loss = self.ce_loss_fn(outputs, labels)
        self.cumm_ce_loss += float(ce_loss)
        return ce_loss

    def _batch_eval(
            self,
            features: torch.Tensor,
            labels: torch.Tensor,
            predictions: torch.Tensor,
            hidden_repr: torch.Tensor,
            ) -> float:

        #predictions = torch.argmax(predictions, dim=-1)
        frequency = ((torch.abs(predictions - labels.to(self.device)) == 0).sum())
        return float(frequency)
